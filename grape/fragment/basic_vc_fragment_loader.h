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

#ifndef GRAPE_FRAGMENT_BASIC_VC_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_BASIC_VC_FRAGMENT_LOADER_H_

#include "grape/utils/memory_tracker.h"

namespace grape {

template <typename FRAG_T>
class BasicVCFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;
  using edge_t = typename fragment_t::edge_t;
  using vertices_t = typename fragment_t::vertices_t;

  static constexpr size_t shuffle_out_size = 4096000;

 public:
  explicit BasicVCFragmentLoader(const CommSpec& comm_spec, int64_t vnum,
                                 int load_concurrency)
      : comm_spec_(comm_spec),
        partitioner_(comm_spec.fnum(), vnum),
        bucket_num_(load_concurrency),
        vnum_(vnum),
        load_concurrency_(load_concurrency) {
    comm_spec_.Dup();
    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag, shuffle_out_size);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag_[fid].DisableComm();
      }
    }

#ifdef TRACKING_MEMORY_ALLOCATIONS
    // allocate shuffle out buffers
    MemoryTracker::GetInstance().allocate(
        (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) * comm_spec_.fnum() *
        shuffle_out_size);
#endif
    edge_recv_thread_ =
        std::thread(&BasicVCFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;

    auto src_vertices = partitioner_.get_src_vertices(comm_spec_.fid());
    src_start_ = src_vertices.begin_value();
    src_chunk_ = (src_vertices.size() + bucket_num_ - 1) / bucket_num_;

    auto dst_vertices = partitioner_.get_dst_vertices(comm_spec_.fid());
    dst_start_ = dst_vertices.begin_value();
    dst_chunk_ = (dst_vertices.size() + bucket_num_ - 1) / bucket_num_;
  }

  ~BasicVCFragmentLoader() {
    if (recv_thread_running_) {
      for (auto& e : edges_to_frag_) {
        e.Flush();
      }
      edge_recv_thread_.join();
    }
  }

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t fid = partitioner_.get_edge_partition(src, dst);
    edges_to_frag_[fid].Emplace(src, dst, data);
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) {
    for (auto& e : edges_to_frag_) {
      e.Flush();
    }
    edge_recv_thread_.join();
    recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());
    got_edges_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].buffers()));

    size_t got_edges_memory_usage = 0;
    for (auto& buf : got_edges_) {
      got_edges_memory_usage +=
          buf.size() * (sizeof(oid_t) * 2 + sizeof(edata_t));
    }
#ifdef TRACKING_MEMORY_ALLOCATIONS
    // allocate edges recv buffer
    MemoryTracker::GetInstance().allocate(got_edges_memory_usage);
#endif

    edges_to_frag_[comm_spec_.fid()].Clear();
#ifdef TRACKING_MEMORY_ALLOCATIONS
    // deallocate shuffle out buffers
    MemoryTracker::GetInstance().deallocate(
        (sizeof(oid_t) + sizeof(oid_t) + sizeof(edata_t)) * comm_spec_.fnum() *
        shuffle_out_size);
#endif

    std::vector<size_t> bucket_edge_num(bucket_num_ * bucket_num_, 0);
    {
      std::vector<std::vector<size_t>> thread_local_bucket_edge_num(
          load_concurrency_);
      std::vector<std::thread> scan_threads;
      for (int i = 0; i < load_concurrency_; ++i) {
        scan_threads.emplace_back(
            [this, &bucket_edge_num, &thread_local_bucket_edge_num](int tid) {
              auto& vec = thread_local_bucket_edge_num[tid];
              vec.clear();
              vec.resize(bucket_num_ * bucket_num_, 0);

              for (auto& buffer : got_edges_) {
                size_t size = buffer.size();
                size_t chunk =
                    (size + load_concurrency_ - 1) / load_concurrency_;
                size_t start = std::min(tid * chunk, size);
                size_t end = std::min(start + chunk, size);
                range_foreach_helper(
                    buffer, start, end,
                    [&vec, this](const oid_t& src, const oid_t& dst) {
                      ++vec[edgeBucketId(src, dst)];
                    },
                    make_index_sequence<2>{});
              }
            },
            i);
      }
      for (auto& thrd : scan_threads) {
        thrd.join();
      }
      for (auto& vec : thread_local_bucket_edge_num) {
        for (size_t i = 0; i < bucket_edge_num.size(); ++i) {
          bucket_edge_num[i] += vec[i];
        }
      }
    }

    std::vector<std::atomic<size_t>> bucket_edge_cursor(bucket_num_ *
                                                        bucket_num_);
    std::vector<size_t> bucket_edge_offset;
    size_t edge_num = 0;
    for (size_t i = 0; i < bucket_num_ * bucket_num_; ++i) {
      bucket_edge_cursor[i].store(edge_num);
      bucket_edge_offset.emplace_back(edge_num);
      edge_num += bucket_edge_num[i];
    }
    bucket_edge_offset.emplace_back(edge_num);

    std::vector<edge_t> edges;
    edges.resize(edge_num);
#ifdef TRACKING_MEMORY_ALLOCATIONS
    // allocate edges buffer
    MemoryTracker::GetInstance().allocate(sizeof(edge_t) * edge_num);
#endif

    {
      static constexpr size_t thread_local_cache_size = 128;
#ifdef TRACKING_MEMORY_ALLOCATIONS
      // allocate thread local cache
      MemoryTracker::GetInstance().allocate(
          sizeof(edge_t) * bucket_num_ * bucket_num_ * thread_local_cache_size *
          load_concurrency_);
#endif
      std::vector<std::thread> insert_threads;
      for (int i = 0; i < load_concurrency_; ++i) {
        insert_threads.emplace_back(
            [this, &bucket_edge_cursor, &edges](int tid) {
              std::vector<std::vector<edge_t>> thread_local_edges(bucket_num_ *
                                                                  bucket_num_);

              for (auto& buffer : got_edges_) {
                size_t size = buffer.size();
                size_t chunk =
                    (size + load_concurrency_ - 1) / load_concurrency_;
                size_t start = std::min(tid * chunk, size);
                size_t end = std::min(start + chunk, size);
                range_foreach_rval(
                    buffer, start, end,
                    [&, this](oid_t&& src, oid_t&& dst, edata_t&& data) {
                      int idx = edgeBucketId(src, dst);
                      thread_local_edges[idx].emplace_back(
                          std::move(src), std::move(dst), std::move(data));

                      if (thread_local_edges[idx].size() ==
                          thread_local_cache_size) {
                        size_t cursor = bucket_edge_cursor[idx].fetch_add(
                            thread_local_cache_size);
                        std::copy(thread_local_edges[idx].begin(),
                                  thread_local_edges[idx].end(),
                                  edges.begin() + cursor);
                        thread_local_edges[idx].clear();
                      }
                    });
              }

              for (size_t i = 0; i < bucket_num_ * bucket_num_; ++i) {
                if (!thread_local_edges[i].empty()) {
                  size_t cursor = bucket_edge_cursor[i].fetch_add(
                      thread_local_edges[i].size());
                  std::copy(thread_local_edges[i].begin(),
                            thread_local_edges[i].end(),
                            edges.begin() + cursor);
                }
              }
            },
            i);
      }
      for (auto& thrd : insert_threads) {
        thrd.join();
      }
#ifdef TRACKING_MEMORY_ALLOCATIONS
      // deallocate thread local cache
      MemoryTracker::GetInstance().deallocate(
          sizeof(edge_t) * bucket_num_ * bucket_num_ * thread_local_cache_size *
          load_concurrency_);
#endif
      got_edges_.clear();
#ifdef TRACKING_MEMORY_ALLOCATIONS
      // deallocate edges recv buffer
      MemoryTracker::GetInstance().deallocate(got_edges_memory_usage);
#endif
    }

    fragment.reset(new fragment_t());
    fragment->Init(comm_spec_, vnum_, std::move(edges), bucket_num_,
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
      got_edges_.emplace_back(std::move(data_in.buffers()));
      data_in.Clear();
    }
  }

  int edgeBucketId(const oid_t& src, const oid_t& dst) const {
    int src_bucket_id = (src - src_start_) / src_chunk_;
    int dst_bucket_id = (dst - dst_start_) / dst_chunk_;
    return src_bucket_id * bucket_num_ + dst_bucket_id;
  }

  CommSpec comm_spec_;
  VCPartitioner<oid_t> partitioner_;
  int bucket_num_;
  oid_t src_start_;
  oid_t dst_start_;
  oid_t src_chunk_;
  oid_t dst_chunk_;

  int64_t vnum_;
  int load_concurrency_;

  std::vector<ShuffleBufferTuple<oid_t, oid_t, edata_t>> got_edges_;

  std::vector<ShuffleOut<oid_t, oid_t, edata_t>> edges_to_frag_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  static constexpr int edge_tag = 6;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_VC_FRAGMENT_LOADER_H_
