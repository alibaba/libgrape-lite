/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPE_FRAGMENT_BASIC_VC_DS_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_BASIC_VC_DS_FRAGMENT_LOADER_H_

#include "grape/utils/memory_tracker.h"

namespace grape {

template <typename OID_T>
class VCEdgeBucketer {
  struct bucket_info {
    OID_T src_start;
    OID_T src_chunk;
    OID_T dst_start;
    OID_T dst_chunk;
  };

 public:
  VCEdgeBucketer(const VCPartitioner<OID_T>& partitioner, int bucket_num)
      : partitioner_(partitioner), bucket_num_(bucket_num) {
    fid_t fnum = partitioner_.fnum();
    for (fid_t i = 0; i < fnum; ++i) {
      auto src_vertices = partitioner_.get_src_vertices(i);
      OID_T src_start = src_vertices.begin_value();
      OID_T src_chunk = (src_vertices.size() + bucket_num - 1) / bucket_num;

      auto dst_vertices = partitioner_.get_dst_vertices(i);
      OID_T dst_start = dst_vertices.begin_value();
      OID_T dst_chunk = (dst_vertices.size() + bucket_num - 1) / bucket_num;

      bucket_infos_.emplace_back(
          bucket_info{src_start, src_chunk, dst_start, dst_chunk});
    }
  }

  int get_bucket_id(const OID_T& src, const OID_T& dst) const {
    fid_t fid = partitioner_.get_edge_partition(src, dst);
    auto& info = bucket_infos_[fid];
    int src_bucket_id = (src - info.src_start) / info.src_chunk;
    int dst_bucket_id = (dst - info.dst_start) / info.dst_chunk;
    return src_bucket_id * bucket_num_ + dst_bucket_id;
  }

  std::pair<fid_t, int> get_partition_bucket_id(const OID_T& src,
                                                const OID_T& dst) const {
    fid_t fid = partitioner_.get_edge_partition(src, dst);
    auto& info = bucket_infos_[fid];
    int src_bucket_id = (src - info.src_start) / info.src_chunk;
    int dst_bucket_id = (dst - info.dst_start) / info.dst_chunk;
    return std::make_pair(fid, src_bucket_id * bucket_num_ + dst_bucket_id);
  }

 private:
  const VCPartitioner<OID_T>& partitioner_;
  int bucket_num_;
  std::vector<bucket_info> bucket_infos_;
};

template <typename FRAG_T>
class BasicVCDSFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;
  using edge_t = typename fragment_t::edge_t;
  using vertices_t = typename fragment_t::vertices_t;

  static constexpr size_t shuffle_out_size = 4096000;

 public:
  explicit BasicVCDSFragmentLoader(const CommSpec& comm_spec, int64_t vnum,
                                   int load_concurrency)
      : comm_spec_(comm_spec),
        partitioner_(comm_spec.fnum(), vnum),
        bucketer_(nullptr),
        vnum_(vnum),
        load_concurrency_(load_concurrency) {
    comm_spec_.Dup();

    MPI_Allreduce(&load_concurrency_, &bucket_num_, 1, MPI_INT, MPI_MIN,
                  comm_spec_.comm());
    bucketer_ = std::unique_ptr<VCEdgeBucketer<oid_t>>(
        new VCEdgeBucketer<oid_t>(partitioner_, bucket_num_));

    partition_bucket_edge_num_.resize(comm_spec_.fnum());
    for (auto& vec : partition_bucket_edge_num_) {
      vec.resize(bucket_num_ * bucket_num_, 0);
    }
  }

  ~BasicVCDSFragmentLoader() {
    if (recv_thread_running_) {
      for (auto& e : edges_to_frag_) {
        e.Flush();
      }
      edge_recv_thread_.join();
      got_edges_.DecProducerNum();
      for (auto& thrd : edge_move_threads_) {
        thrd.join();
      }
    }
  }

  void RecordEdge(const oid_t& src, const oid_t& dst) {
    auto pair = bucketer_->get_partition_bucket_id(src, dst);
    ++partition_bucket_edge_num_[pair.first][pair.second];
  }

  void AllocateBuffers() {
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      if (comm_spec_.FragToWorker(fid) == comm_spec_.worker_id()) {
        MPI_Reduce(MPI_IN_PLACE, partition_bucket_edge_num_[fid].data(),
                   partition_bucket_edge_num_[fid].size(), MPI_LONG_LONG_INT,
                   MPI_SUM, comm_spec_.FragToWorker(fid), comm_spec_.comm());
      } else {
        MPI_Reduce(partition_bucket_edge_num_[fid].data(), nullptr,
                   partition_bucket_edge_num_[fid].size(), MPI_LONG_LONG_INT,
                   MPI_SUM, comm_spec_.FragToWorker(fid), comm_spec_.comm());
      }
    }
    size_t edge_num = 0;
    bucket_cursor_ =
        std::vector<std::atomic<size_t>>(bucket_num_ * bucket_num_);
    for (int k = 0; k < bucket_num_ * bucket_num_; ++k) {
      bucket_cursor_[k].store(edge_num);
      edge_num += partition_bucket_edge_num_[comm_spec_.fid()][k];
    }
    edges_.resize(edge_num);
#ifdef TRACKING_MEMORY
    // allocate memory for edges
    MemoryTracker::GetInstance().allocate((sizeof(edge_t)) * edge_num);
#endif

    got_edges_.SetProducerNum(2);

    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag, shuffle_out_size);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag_[fid].DisableComm();
      }
    }
#ifdef TRACKING_MEMORY
    // allocate memory for edge shuffle-out buffer
    MemoryTracker::GetInstance().allocate(
        (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) * comm_spec_.fnum() *
        shuffle_out_size);
#endif
    static constexpr size_t thread_local_cache_size = 128;
    for (int i = 0; i < load_concurrency_; ++i) {
      edge_move_threads_.emplace_back([this] {
        ShuffleBufferTuple<oid_t, oid_t, edata_t> cur;
        std::vector<std::vector<edge_t>> edges_cache(bucket_num_ * bucket_num_);
#ifdef TRACKING_MEMORY
        // allocate memory for thread local edge cache
        MemoryTracker::GetInstance().allocate(
            (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) *
            thread_local_cache_size);
#endif
        while (got_edges_.Get(cur)) {
          size_t cur_size = cur.size();
          foreach_rval(cur, [&edges_cache, this](oid_t src, oid_t dst,
                                                 edata_t data) {
            int bucket_id = bucketer_->get_bucket_id(src, dst);
            edges_cache[bucket_id].emplace_back(src, dst, data);
            if (edges_cache[bucket_id].size() >= thread_local_cache_size) {
              size_t cursor = bucket_cursor_[bucket_id].fetch_add(
                  edges_cache[bucket_id].size());
              std::copy(edges_cache[bucket_id].begin(),
                        edges_cache[bucket_id].end(), edges_.begin() + cursor);
              edges_cache[bucket_id].clear();
            }
          });
#ifdef TRACKING_MEMORY
          // deallocate edge shuffle-in buffer
          MemoryTracker::GetInstance().deallocate(
              (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) * cur_size);
#endif
        }
        for (int i = 0; i < bucket_num_ * bucket_num_; ++i) {
          if (!edges_cache[i].empty()) {
            size_t cursor = bucket_cursor_[i].fetch_add(edges_cache[i].size());
            std::copy(edges_cache[i].begin(), edges_cache[i].end(),
                      edges_.begin() + cursor);
          }
        }
#ifdef TRACKING_MEMORY
        // deallocate memory for thread local edge cache
        MemoryTracker::GetInstance().deallocate(
            (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) *
            thread_local_cache_size);
#endif
      });
    }
    edge_recv_thread_ =
        std::thread(&BasicVCDSFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;
  }

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t fid = partitioner_.get_edge_partition(src, dst);
    edges_to_frag_[fid].Emplace(src, dst, data);
    if (fid == comm_spec_.fid() &&
        edges_to_frag_[fid].buffers().size() >= shuffle_out_size) {
#ifdef TRACKING_MEMORY
      // allocate memory for edge shuffle-out(to self) buffer
      MemoryTracker::GetInstance().allocate(
          (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) *
          edges_to_frag_[fid].buffers().size());
#endif
      got_edges_.Put(std::move(edges_to_frag_[fid].buffers()));
      edges_to_frag_[fid].Clear();
    }
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) {
    for (auto& e : edges_to_frag_) {
      e.Flush();
    }
    got_edges_.Put(std::move(edges_to_frag_[comm_spec_.fid()].buffers()));
    edge_recv_thread_.join();
    recv_thread_running_ = false;
    edges_to_frag_.clear();
#ifdef TRACKING_MEMORY
    // deallocate memory for edge shuffle-out buffer
    MemoryTracker::GetInstance().deallocate(
        (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) * comm_spec_.fnum() *
        shuffle_out_size);
#endif
    got_edges_.DecProducerNum();
    for (auto& thrd : edge_move_threads_) {
      thrd.join();
    }

    MPI_Barrier(comm_spec_.comm());

    std::vector<size_t> bucket_edge_offset;
    size_t en = 0;
    for (size_t i = 0; i < bucket_num_ * bucket_num_; ++i) {
      bucket_edge_offset.emplace_back(en);
      en += partition_bucket_edge_num_[comm_spec_.fid()][i];
    }
    bucket_edge_offset.emplace_back(en);

    fragment.reset(new fragment_t());
    fragment->Init(comm_spec_, vnum_, std::move(edges_), bucket_num_,
                   std::move(bucket_edge_offset));
  }

 private:
  void edgeRecvRoutine() {
    ShuffleIn<oid_t, oid_t, edata_t> data_in;
    data_in.Init(comm_spec_.fnum(), comm_spec_.comm(), edge_tag);
    fid_t dst_fid;
    int src_worker_id;
    while (!data_in.Finished()) {
      src_worker_id = data_in.Recv(dst_fid);
      if (src_worker_id == -1) {
        break;
      }
      CHECK_EQ(dst_fid, comm_spec_.fid());
#ifdef TRACKING_MEMORY
      // allocate memory for edge shuffle-in buffer
      MemoryTracker::GetInstance().allocate(
          (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) *
          data_in.buffers().size());
#endif
      got_edges_.Put(std::move(data_in.buffers()));
      data_in.Clear();
    }
    got_edges_.DecProducerNum();
  }

  std::vector<std::vector<size_t>> partition_bucket_edge_num_;

  CommSpec comm_spec_;
  VCPartitioner<oid_t> partitioner_;
  int bucket_num_;
  std::unique_ptr<VCEdgeBucketer<oid_t>> bucketer_;

  int64_t vnum_;
  int load_concurrency_;

  std::vector<std::atomic<size_t>> bucket_cursor_;
  std::vector<edge_t> edges_;

  std::vector<ShuffleOut<oid_t, oid_t, edata_t>> edges_to_frag_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  BlockingQueue<ShuffleBufferTuple<oid_t, oid_t, edata_t>> got_edges_;
  std::vector<std::thread> edge_move_threads_;

  static constexpr int edge_tag = 6;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_VC_DS_FRAGMENT_LOADER_H_
