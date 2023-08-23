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

#ifndef GRAPE_FRAGMENT_REBALANCER_H_
#define GRAPE_FRAGMENT_REBALANCER_H_

#include <stddef.h>

#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "grape/communication/shuffle.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/types.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief Rebalancer aims to rebalance the edges between fragments. For some
 * skewed graphs, hash partitioning on vertex_ids may leads to imbalanced
 * workload in workers. Rebalancer tries to reallocate some vertices and edges
 * in these cases.
 *
 * @tparam FRAG_T
 * @tparam void
 */
template <typename FRAG_T, typename Enable = void>
class Rebalancer;

template <typename FRAG_T>
class Rebalancer<
    FRAG_T, typename std::enable_if<!std::is_same<typename FRAG_T::vdata_t,
                                                  EmptyType>::value>::type> {
  using vertex_t = typename FRAG_T::vertex_t;
  using edge_t = typename FRAG_T::edge_t;
  using vertex_map_t = typename FRAG_T::vertex_map_t;

  static constexpr LoadStrategy load_strategy = FRAG_T::load_strategy;

 public:
  explicit Rebalancer(const CommSpec& comm_spec) {
    LOG(FATAL) << "Not implemented...";
  }

  ~Rebalancer() = default;

  void Rebalance(std::shared_ptr<vertex_map_t> vm_ptr,
                 std::vector<vertex_t>& vertices, std::vector<edge_t>& edges) {
    LOG(FATAL) << "Not implemented...";
  }
};

template <typename FRAG_T>
class Rebalancer<
    FRAG_T, typename std::enable_if<std::is_same<typename FRAG_T::vdata_t,
                                                 EmptyType>::value>::type> {
  using vertex_t = typename FRAG_T::vertex_t;
  using edge_t = typename FRAG_T::edge_t;
  using vertex_map_t = typename FRAG_T::vertex_map_t;
  using vid_t = typename FRAG_T::vid_t;
  using edata_t = typename FRAG_T::edata_t;

  static constexpr LoadStrategy load_strategy = FRAG_T::load_strategy;

 public:
  explicit Rebalancer(const CommSpec& comm_spec, size_t rebalance_vertex_factor)
      : comm_spec_(comm_spec),
        rebalance_vertex_factor_(rebalance_vertex_factor) {}
  ~Rebalancer() {}

  void Rebalance(std::shared_ptr<vertex_map_t> vm_ptr,
                 std::vector<vertex_t>& vertices, std::vector<edge_t>& edges) {
    initDegreeLists(vm_ptr, edges);
    generateNewPartition(vm_ptr);
    vm_ptr->UpdateToBalance(vnum_list_, gid_maps_);
    reprocessEdges(vm_ptr, edges);
  }

 private:
  void initDegreeLists(std::shared_ptr<vertex_map_t> vm_ptr,
                       std::vector<edge_t>& edges) {
    degree_lists_.clear();
    degree_lists_.resize(comm_spec_.fnum());

    {
      fid_t fid = comm_spec_.fid();
      auto& degree_list = degree_lists_[fid];
      degree_list.clear();
      degree_list.resize(vm_ptr->GetInnerVertexSize(fid), 0);
      if (load_strategy == LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          if (vm_ptr->GetFidFromGid(e.src_) == fid) {
            ++degree_list[vm_ptr->GetLidFromGid(e.src_)];
          }
        }
      } else if (load_strategy == LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          if (vm_ptr->GetFidFromGid(e.dst_) == fid) {
            ++degree_list[vm_ptr->GetLidFromGid(e.dst_)];
          }
        }
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (auto& e : edges) {
          if (vm_ptr->GetFidFromGid(e.src_) == fid) {
            ++degree_list[vm_ptr->GetLidFromGid(e.src_)];
          }
          if (vm_ptr->GetFidFromGid(e.dst_) == fid) {
            ++degree_list[vm_ptr->GetLidFromGid(e.dst_)];
          }
        }
      }
    }
    {
      int worker_id = comm_spec_.worker_id();
      int worker_num = comm_spec_.worker_num();
      std::thread recv_thread([&]() {
        int src_worker_id = (worker_id + 1) % worker_num;
        while (src_worker_id != worker_id) {
          fid_t src_fid = comm_spec_.WorkerToFrag(src_worker_id);
          RecvVector(degree_lists_[src_fid], src_worker_id, comm_spec_.comm(),
                     degree_tag);
          src_worker_id = (src_worker_id + 1) % worker_num;
        }
      });
      std::thread send_thread([&]() {
        int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
        while (dst_worker_id != worker_id) {
          SendVector(degree_lists_[comm_spec_.fid()], dst_worker_id,
                     comm_spec_.comm(), degree_tag);
          dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
        }
      });
      send_thread.join();
      recv_thread.join();
    }
  }

  void generateNewPartition(std::shared_ptr<vertex_map_t> vm_ptr) {
    size_t total_enum = 0;
    for (auto& dl : degree_lists_) {
      for (auto d : dl) {
        total_enum += d;
      }
    }
    vid_t total_vnum = 0;
    for (auto& dl : degree_lists_) {
      total_vnum += static_cast<vid_t>(dl.size());
    }

    vnum_list_.clear();
    fid_t fnum = comm_spec_.fnum();
    vnum_list_.resize(fnum);

    size_t remaining_edges =
        total_enum + static_cast<size_t>(total_vnum) * rebalance_vertex_factor_;
    vid_t remaining_vertices = total_vnum;

    fid_t cur_part_id = 0;
    vid_t cur_local_id = 0;
    for (fid_t i = 0; i < fnum; ++i) {
      fid_t remaining_part_num = fnum - i;
      size_t expected_edge_num = remaining_edges / remaining_part_num;
      if (remaining_part_num == 1) {
        CHECK(remaining_vertices <= vm_ptr->MaxVertexNum());
        vnum_list_[i] = remaining_vertices;
      } else {
        size_t got_edges = 0;
        vid_t got_vertices = 0;
        while (cur_part_id != fnum) {
          if (cur_local_id ==
              static_cast<vid_t>(degree_lists_[cur_part_id].size())) {
            cur_local_id = 0;
            ++cur_part_id;
          } else {
            got_edges += (degree_lists_[cur_part_id][cur_local_id] +
                          rebalance_vertex_factor_);
            if (got_edges > expected_edge_num) {
              CHECK(got_vertices <= vm_ptr->MaxVertexNum());
              vnum_list_[i] = got_vertices;
              remaining_vertices -= got_vertices;
              break;
            }
            remaining_edges -= (degree_lists_[cur_part_id][cur_local_id] +
                                rebalance_vertex_factor_);
            ++cur_local_id;
            ++got_vertices;
          }
        }
      }
    }

    gid_maps_.clear();
    gid_maps_.resize(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      gid_maps_[i].resize(degree_lists_[i].size());
    }

    cur_part_id = 0;
    cur_local_id = 0;
    for (fid_t i = 0; i < fnum; ++i) {
      for (vid_t j = 0; j < vnum_list_[i]; ++j) {
        if (cur_local_id ==
            static_cast<vid_t>(degree_lists_[cur_part_id].size())) {
          cur_local_id = 0;
          ++cur_part_id;
        }
        gid_maps_[cur_part_id][cur_local_id] = vm_ptr->Lid2Gid(i, j);
        ++cur_local_id;
      }
    }

    degree_lists_.clear();
    degree_lists_.shrink_to_fit();
  }

  void reprocessEdges(std::shared_ptr<vertex_map_t> vm_ptr,
                      std::vector<edge_t>& edges) {
    std::vector<edge_t> edges_to_send;

    {
      size_t evec_size = edges.size();
      size_t count = 0;
      fid_t fid = comm_spec_.fid();
      if (load_strategy == LoadStrategy::kOnlyOut) {
        for (size_t i = 0; i < evec_size; ++i) {
          edge_t e;
          auto& old_e = edges[i];
          e.src_ = gid_maps_[vm_ptr->GetFidFromGid(old_e.src_)]
                            [vm_ptr->GetLidFromGid(old_e.src_)];
          e.dst_ = gid_maps_[vm_ptr->GetFidFromGid(old_e.dst_)]
                            [vm_ptr->GetLidFromGid(old_e.dst_)];
          if (vm_ptr->GetFidFromGid(old_e.src_) == fid) {
            e.set_edata(std::move(old_e.edata_));
            if (vm_ptr->GetFidFromGid(e.src_) == fid) {
              edges[i - count] = std::move(e);
            } else {
              edges_to_send.emplace_back(std::move(e));
              ++count;
            }
          } else {
            ++count;
          }
        }
        edges.resize(evec_size - count);
      } else if (load_strategy == LoadStrategy::kOnlyIn) {
        for (size_t i = 0; i < evec_size; ++i) {
          edge_t e;
          auto& old_e = edges[i];
          e.src_ = gid_maps_[vm_ptr->GetFidFromGid(old_e.src_)]
                            [vm_ptr->GetLidFromGid(old_e.src_)];
          e.dst_ = gid_maps_[vm_ptr->GetFidFromGid(old_e.dst_)]
                            [vm_ptr->GetLidFromGid(old_e.dst_)];
          if (vm_ptr->GetFidFromGid(old_e.dst_) == fid) {
            e.set_edata(std::move(old_e.edata_));
            if (vm_ptr->GetFidFromGid(e.dst_) == fid) {
              edges[i - count] = std::move(e);
            } else {
              edges_to_send.emplace_back(std::move(e));
              ++count;
            }
          } else {
            ++count;
          }
        }
        edges.resize(evec_size - count);
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (size_t i = 0; i < evec_size; ++i) {
          edge_t e;
          auto& old_e = edges[i];
          fid_t old_src_fid = vm_ptr->GetFidFromGid(old_e.src_);
          fid_t old_dst_fid = vm_ptr->GetFidFromGid(old_e.dst_);
          e.src_ = gid_maps_[old_src_fid][vm_ptr->GetLidFromGid(old_e.src_)];
          e.dst_ = gid_maps_[old_dst_fid][vm_ptr->GetLidFromGid(old_e.dst_)];
          fid_t new_src_fid = vm_ptr->GetFidFromGid(e.src_);
          fid_t new_dst_fid = vm_ptr->GetFidFromGid(e.dst_);
          if (new_src_fid == old_src_fid && new_dst_fid == old_dst_fid) {
            e.set_edata(std::move(old_e.edata_));
            edges[i - count] = std::move(e);
          } else if (old_src_fid == fid) {
            e.set_edata(std::move(old_e.edata_));
            edges_to_send.emplace_back(std::move(e));
            ++count;
          } else {
            ++count;
          }
        }
        edges.resize(evec_size - count);
      }
    }

    gid_maps_.clear();
    gid_maps_.shrink_to_fit();

    std::vector<ShuffleOutTriple<vid_t, vid_t, edata_t>> delta_edges_to_frag(
        comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      delta_edges_to_frag[fid].Init(comm_spec_.comm(), delta_edge_tag);
      delta_edges_to_frag[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        delta_edges_to_frag[fid].DisableComm();
      }
    }

    std::thread deltaEdgesRecvThread([&]() {
      ShuffleInTriple<vid_t, vid_t, edata_t> data_in(comm_spec_.fnum() - 1);
      data_in.Init(comm_spec_.comm(), delta_edge_tag);
      int src_worker_id;
      fid_t dst_fid;
      while (!data_in.Finished()) {
        src_worker_id = data_in.Recv(dst_fid);
        if (src_worker_id == -1) {
          break;
        }
        CHECK_EQ(dst_fid, comm_spec_.fid());
        size_t to_append = data_in.Buffer0().size();
        size_t old_size = edges.size();
        edges.resize(old_size + to_append);
        auto* buf_ptr = &edges[old_size];
        auto src_iter = data_in.Buffer0().begin();
        auto dst_iter = data_in.Buffer1().begin();
        auto data_iter = data_in.Buffer2().begin();
        auto src_end = data_in.Buffer0().end();
        while (src_iter != src_end) {
          buf_ptr->SetEndpoint(*src_iter, *dst_iter);
          buf_ptr->set_edata(*data_iter);
          ++src_iter;
          ++dst_iter;
          ++data_iter;
          ++buf_ptr;
        }
      }
    });

    if (load_strategy == LoadStrategy::kOnlyOut) {
      for (auto& e : edges_to_send) {
        fid_t src_fid = vm_ptr->GetFidFromGid(e.src_);
        delta_edges_to_frag[src_fid].Emplace(e.src_, e.dst_, edata_t(e.edata_));
      }
    } else if (load_strategy == LoadStrategy::kOnlyIn) {
      for (auto& e : edges_to_send) {
        fid_t dst_fid = vm_ptr->GetFidFromGid(e.dst_);
        delta_edges_to_frag[dst_fid].Emplace(e.src_, e.dst_, edata_t(e.edata_));
      }
    } else if (load_strategy == LoadStrategy::kBothOutIn) {
      for (auto& e : edges_to_send) {
        fid_t src_fid = vm_ptr->GetFidFromGid(e.src_);
        fid_t dst_fid = vm_ptr->GetFidFromGid(e.dst_);
        delta_edges_to_frag[src_fid].Emplace(e.src_, e.dst_, edata_t(e.edata_));
        if (src_fid != dst_fid) {
          delta_edges_to_frag[dst_fid].Emplace(e.src_, e.dst_,
                                               edata_t(e.edata_));
        }
      }
    }

    edges_to_send.clear();

    for (auto& de : delta_edges_to_frag) {
      de.Flush();
    }

    deltaEdgesRecvThread.join();

    {
      fid_t fid = comm_spec_.fid();
      size_t to_append = delta_edges_to_frag[fid].Buffer0().size();
      size_t old_size = edges.size();
      edges.resize(old_size + to_append);
      auto* buf_ptr = &edges[old_size];
      auto src_iter = delta_edges_to_frag[fid].Buffer0().begin();
      auto dst_iter = delta_edges_to_frag[fid].Buffer1().begin();
      auto data_iter = delta_edges_to_frag[fid].Buffer2().begin();
      auto src_end = delta_edges_to_frag[fid].Buffer0().end();
      while (src_iter != src_end) {
        buf_ptr->SetEndpoint(*src_iter, *dst_iter);
        buf_ptr->set_edata(*data_iter);
        ++src_iter;
        ++dst_iter;
        ++data_iter;
        ++buf_ptr;
      }
    }
  }

 private:
  CommSpec comm_spec_;
  const size_t rebalance_vertex_factor_;

  std::vector<std::vector<vid_t>> degree_lists_;
  std::vector<vid_t> vnum_list_;
  std::vector<std::vector<vid_t>> gid_maps_;

  static constexpr int degree_tag = 7;
  static constexpr int delta_edge_tag = 8;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_REBALANCER_H_
