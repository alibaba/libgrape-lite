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

#ifndef GRAPE_FRAGMENT_BASIC_EFILE_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_BASIC_EFILE_FRAGMENT_LOADER_H_

#include <atomic>

#include "grape/communication/shuffle.h"
#include "grape/fragment/basic_fragment_loader_base.h"
#include "grape/fragment/rebalancer.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/vertex_map/vertex_map.h"

namespace grape {

inline size_t custom_hash(size_t val) {
  val = (val ^ 61) ^ (val >> 16);
  val = val + (val << 3);
  val = val ^ (val >> 4);
  val = val * 0x27d4eb2d;
  val = val ^ (val >> 15);
  return val;
}

template <typename FRAG_T>
class BasicEFileFragmentLoader : public BasicFragmentLoaderBase<FRAG_T> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using internal_oid_t = typename InternalOID<oid_t>::type;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

 public:
  explicit BasicEFileFragmentLoader(const CommSpec& comm_spec,
                                    const LoadGraphSpec& spec)
      : BasicFragmentLoaderBase<FRAG_T>(comm_spec, spec) {
    if (spec_.partitioner_type != PartitionerType::kHashPartitioner) {
      LOG(ERROR) << "Only hash partitioner is supported in "
                    "BasicEFileFragmentLoader";
      spec_.partitioner_type = PartitionerType::kHashPartitioner;
    }
    if (spec_.rebalance) {
      LOG(ERROR) << "Rebalance is not supported in BasicEFileFragmentLoader";
      spec_.rebalance = false;
    }
    partitioner_ = std::unique_ptr<HashPartitioner<oid_t>>(
        new HashPartitioner<oid_t>(comm_spec_.fnum()));
    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag, 4096000);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag_[fid].DisableComm();
      }
    }

    edge_recv_thread_ =
        std::thread(&BasicEFileFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;

    concurrency_ = spec.load_concurrency;
  }

  ~BasicEFileFragmentLoader() {
    if (recv_thread_running_) {
      for (auto& ea : edges_to_frag_) {
        ea.Flush();
      }
      edge_recv_thread_.join();
    }
  }

  void AddVertex(const oid_t& id, const vdata_t& data) override {}

  void ConstructVertices() override {}

  void AddEdge(const oid_t& src, const oid_t& dst,
               const edata_t& data) override {
    internal_oid_t internal_src(src);
    internal_oid_t internal_dst(dst);
    fid_t src_fid = partitioner_->GetPartitionId(internal_src);
    fid_t dst_fid = partitioner_->GetPartitionId(internal_dst);
    if (src_fid == comm_spec_.fnum() || dst_fid == comm_spec_.fnum()) {
      LOG(ERROR) << "Unknown partition id for edge " << src << " -> " << dst;
    } else {
      edges_to_frag_[src_fid].Emplace(internal_src, internal_dst, data);
      if (src_fid != dst_fid) {
        edges_to_frag_[dst_fid].Emplace(internal_src, internal_dst, data);
      }
    }
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) override {
    for (auto& ea : edges_to_frag_) {
      ea.Flush();
    }

    edge_recv_thread_.join();
    recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());
    got_edges_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].buffers()));
    edges_to_frag_[comm_spec_.fid()].Clear();

    double t0 = -grape::GetCurrentTime();
    std::unique_ptr<VertexMap<oid_t, vid_t>> vm_ptr(
        new VertexMap<oid_t, vid_t>());
    {
      VertexMapBuilder<oid_t, vid_t> builder(
          comm_spec_.fid(), comm_spec_.fnum(), std::move(partitioner_),
          spec_.idxer_type);
      if (concurrency_ == 1) {
        for (auto& buffers : got_edges_) {
          foreach_helper(
              buffers,
              [&builder](const internal_oid_t& src, const internal_oid_t& dst) {
                builder.add_vertex(src);
                builder.add_vertex(dst);
              },
              make_index_sequence<2>{});
        }
      } else {
        std::atomic<size_t> idx(0);
        std::vector<std::vector<internal_oid_t>> vertices(concurrency_);
        std::vector<std::vector<internal_oid_t>> vertices_mat(concurrency_ *
                                                              concurrency_);
        std::vector<std::thread> threads;
        std::vector<double> thread_time(concurrency_, 0);
        for (int i = 0; i < concurrency_; ++i) {
          threads.emplace_back(
              [&, this](int tid) {
                double tt = -grape::GetCurrentTime();
                fid_t fid = comm_spec_.fid();
                // auto& vec = vertices[tid];
                for (auto& buffer : got_edges_) {
                  size_t size = buffer.size();
                  size_t chunk = (size + concurrency_ - 1) / concurrency_;
                  size_t start = std::min(size, chunk * tid);
                  size_t end = std::min(size, start + chunk);
                  if (spec_.idxer_type == IdxerType::kLocalIdxer) {
                    range_foreach_helper(
                        buffer, start, end,
                        [&](const internal_oid_t& src,
                            const internal_oid_t& dst) {
                          int src_hash =
                              custom_hash(std::hash<internal_oid_t>()(src)) %
                              concurrency_;
                          vertices_mat[tid * concurrency_ + src_hash]
                              .emplace_back(src);
                          int dst_hash =
                              custom_hash(std::hash<internal_oid_t>()(dst)) %
                              concurrency_;
                          vertices_mat[tid * concurrency_ + dst_hash]
                              .emplace_back(dst);
                          // vec.emplace_back(src);
                          // vec.emplace_back(dst);
                        },
                        make_index_sequence<2>{});
                  } else {
                    range_foreach_helper(
                        buffer, start, end,
                        [&](const internal_oid_t& src,
                            const internal_oid_t& dst) {
                          if (builder.get_fragment_id(src) == fid) {
                            int src_hash =
                                custom_hash(std::hash<internal_oid_t>()(src)) %
                                concurrency_;
                            vertices_mat[tid * concurrency_ + src_hash]
                                .emplace_back(src);
                            // vec.emplace_back(src);
                          }
                          if (builder.get_fragment_id(dst) == fid) {
                            int dst_hash =
                                custom_hash(std::hash<internal_oid_t>()(dst)) %
                                concurrency_;
                            vertices_mat[tid * concurrency_ + dst_hash]
                                .emplace_back(dst);
                            // vec.emplace_back(dst);
                          }
                        },
                        make_index_sequence<2>{});
                  }
                }
                // DistinctSort(vec);
                tt += grape::GetCurrentTime();
                thread_time[tid] = tt;
              },
              i);
        }
        for (auto& thrd : threads) {
          thrd.join();
        }
        show_thread_timing(thread_time, "parse vertices");
        std::vector<std::thread> aggregate_threads;
        for (int i = 0; i < concurrency_; ++i) {
          aggregate_threads.emplace_back(
              [&, this](int tid) {
                double tt = -grape::GetCurrentTime();
                auto& vec = vertices[tid];
                for (int j = 0; j < concurrency_; ++j) {
                  vec.insert(vec.end(),
                             vertices_mat[j * concurrency_ + tid].begin(),
                             vertices_mat[j * concurrency_ + tid].end());
                }
                DistinctSort(vec);
                tt += grape::GetCurrentTime();
                thread_time[tid] = tt;
              },
              i);
        }
        for (auto& thrd : aggregate_threads) {
          thrd.join();
        }
        show_thread_timing(thread_time, "aggregate vertices");
        // TODO(luoxiaojian): parallelize this part
        double tx = -grape::GetCurrentTime();
        for (auto& vec : vertices) {
          for (auto& v : vec) {
            builder.add_vertex(v);
          }
        }
        tx += grape::GetCurrentTime();
        LOG(INFO) << "[worker-" << comm_spec_.worker_id()
                  << "] finished adding vertices, time: " << tx << " s";
      }
      double ty = -grape::GetCurrentTime();
      builder.finish(comm_spec_, *vm_ptr);
      ty += grape::GetCurrentTime();
      LOG(INFO) << "[worker-" << comm_spec_.worker_id()
                << "] finished building vertex map, time: " << ty << " s";
    }
    MPI_Barrier(comm_spec_.comm());
    t0 += grape::GetCurrentTime();
    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished constructing vertex_map, time: " << t0 << " s";
    }

    double t1 = -grape::GetCurrentTime();
    std::vector<Edge<vid_t, edata_t>> processed_edges;
    if (concurrency_ == 1) {
      for (auto& buffers : got_edges_) {
        foreach_rval(buffers, [&processed_edges, &vm_ptr](internal_oid_t&& src,
                                                          internal_oid_t&& dst,
                                                          edata_t&& data) {
          vid_t src_gid, dst_gid;
          if (vm_ptr->GetGid(oid_t(src), src_gid) &&
              vm_ptr->GetGid(oid_t(dst), dst_gid)) {
            processed_edges.emplace_back(src_gid, dst_gid, std::move(data));
          }
        });
      }
    } else {
      std::vector<size_t> offsets;
      size_t total = 0;
      for (auto& buffers : got_edges_) {
        offsets.emplace_back(total);
        total += buffers.size();
      }
      processed_edges.resize(total);
      std::vector<std::thread> threads;
      std::vector<double> thread_time(concurrency_, 0);
      for (int i = 0; i < concurrency_; ++i) {
        threads.emplace_back(
            [&, this](int tid) {
              double tt = -grape::GetCurrentTime();
              size_t global_offset = 0;
              for (auto& buffer : got_edges_) {
                size_t size = buffer.size();
                size_t chunk = (size + concurrency_ - 1) / concurrency_;
                size_t start = std::min(size, chunk * tid);
                size_t end = std::min(size, start + chunk);
                size_t local_offset = global_offset + start;
                global_offset += size;
                range_foreach_rval(
                    buffer, start, end,
                    [&](internal_oid_t&& src, internal_oid_t&& dst,
                        edata_t&& data) {
                      vid_t src_gid, dst_gid;
                      if (vm_ptr->GetGidFromInternalOid(src, src_gid) &&
                          vm_ptr->GetGidFromInternalOid(dst, dst_gid)) {
                        processed_edges[local_offset++] = Edge<vid_t, edata_t>(
                            src_gid, dst_gid, std::move(data));
                      } else {
                        processed_edges[local_offset++] = Edge<vid_t, edata_t>(
                            std::numeric_limits<vid_t>::max(),
                            std::numeric_limits<vid_t>::max(), std::move(data));
                      }
                    });
              }
              tt += grape::GetCurrentTime();
              thread_time[tid] = tt;
            },
            i);
      }
      for (auto& thrd : threads) {
        thrd.join();
      }
      show_thread_timing(thread_time, "construct edges");
    }
    MPI_Barrier(comm_spec_.comm());
    t1 += grape::GetCurrentTime();
    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished parsing edges, time: " << t1 << " s";
    }

    double t2 = -grape::GetCurrentTime();
    fragment = std::make_shared<fragment_t>();
    std::vector<internal::Vertex<vid_t, vdata_t>> fake_vertices;
    if (concurrency_ == 1) {
      fragment->Init(comm_spec_, spec_.directed, std::move(vm_ptr),
                     fake_vertices, processed_edges);
    } else {
      fragment->ParallelInit(comm_spec_, spec_.directed, std::move(vm_ptr),
                             fake_vertices, processed_edges, concurrency_);
    }
    MPI_Barrier(comm_spec_.comm());
    t2 += grape::GetCurrentTime();
    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished initializing fragment, time: " << t2 << " s";
    }

    if (!std::is_same<EmptyType, vdata_t>::value) {
      this->InitOuterVertexData(fragment);
    }
  }

 private:
  void edgeRecvRoutine() {
    ShuffleIn<internal_oid_t, internal_oid_t, edata_t> data_in;
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

  std::unique_ptr<HashPartitioner<oid_t>> partitioner_;

  std::vector<ShuffleOut<internal_oid_t, internal_oid_t, edata_t>>
      edges_to_frag_;

  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  std::vector<ShuffleBufferTuple<internal_oid_t, internal_oid_t, edata_t>>
      got_edges_;
  int concurrency_;

  using BasicFragmentLoaderBase<FRAG_T>::comm_spec_;
  using BasicFragmentLoaderBase<FRAG_T>::spec_;
  using BasicFragmentLoaderBase<FRAG_T>::id_parser_;

  using BasicFragmentLoaderBase<FRAG_T>::edge_tag;
};

};  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_EFILE_FRAGMENT_LOADER_H_
