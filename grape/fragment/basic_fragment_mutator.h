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

#ifndef GRAPE_FRAGMENT_BASIC_FRAGMENT_MUTATOR_H_
#define GRAPE_FRAGMENT_BASIC_FRAGMENT_MUTATOR_H_

#include <vector>

#include <grape/communication/shuffle.h>
#include <grape/graph/edge.h>
#include <grape/graph/vertex.h>
#include <grape/utils/concurrent_queue.h>
#include <grape/worker/comm_spec.h>

namespace grape {

template <typename ID_T, typename VDATA_T, typename EDATA_T>
struct Mutation {
  std::vector<internal::Vertex<ID_T, VDATA_T>> vertices_to_add;
  std::vector<ID_T> vertices_to_remove;

  std::vector<Edge<ID_T, EDATA_T>> edges_to_add;
  std::vector<std::pair<ID_T, ID_T>> edges_to_remove;

  std::vector<internal::Vertex<ID_T, VDATA_T>> vertices_to_update;
  std::vector<Edge<ID_T, EDATA_T>> edges_to_update;
};

template <typename FRAG_T>
class BasicFragmentMutator {
  using fragment_t = FRAG_T;
  using vertex_map_t = typename FRAG_T::vertex_map_t;
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vdata_t = typename FRAG_T::vdata_t;
  using edata_t = typename FRAG_T::edata_t;
  using mutation_t = Mutation<vid_t, vdata_t, edata_t>;
  static constexpr LoadStrategy load_strategy = FRAG_T::load_strategy;
  using partitioner_t = typename vertex_map_t::partitioner_t;

 public:
  explicit BasicFragmentMutator(const CommSpec& comm_spec,
                                std::shared_ptr<fragment_t> fragment)
      : comm_spec_(comm_spec),
        fragment_(fragment),
        vm_ptr_(fragment->GetVertexMap()) {
    comm_spec_.Dup();
  }

  ~BasicFragmentMutator() = default;

  void SetPartitioner(const partitioner_t& partitioner) {
    vm_ptr_->SetPartitioner(partitioner);
  }

  void SetPartitioner(partitioner_t&& partitioner) {
    vm_ptr_->SetPartitioner(std::move(partitioner));
  }

  std::shared_ptr<fragment_t> MutateFragment() {
    for (auto& shuf : edges_to_add_) {
      shuf.Flush();
    }
    for (auto& shuf : edges_to_remove_) {
      shuf.Flush();
    }
    for (auto& shuf : edges_to_update_) {
      shuf.Flush();
    }
    recv_thread_.join();

    sync_comm::FlatAllGather<vid_t>(local_vertices_to_remove_,
                                    global_vertices_to_remove_,
                                    comm_spec_.comm());
    processToSelfMessages();
    if (!std::is_same<vdata_t, EmptyType>::value) {
      std::vector<oid_t> global_added_vertices_id;
      sync_comm::FlatAllGather<oid_t>(local_added_vertices_id_,
                                      global_added_vertices_id,
                                      comm_spec_.comm());
      extendVertexMap(global_added_vertices_id);
      size_t local_add_vnum = local_vertices_to_add_.size();
      CHECK_EQ(local_add_vnum, local_added_vertices_id_.size());
      for (size_t i = 0; i < local_add_vnum; ++i) {
        auto& oid = local_added_vertices_id_[i];
        CHECK(vm_ptr_->GetGid(oid, local_vertices_to_add_[i].vid));
      }
      sync_comm::FlatAllGather<internal::Vertex<vid_t, vdata_t>>(
          local_vertices_to_add_, global_vertices_to_add_, comm_spec_.comm());
      sync_comm::FlatAllGather<internal::Vertex<vid_t, vdata_t>>(
          local_vertices_to_update_, global_vertices_to_update_,
          comm_spec_.comm());
    } else {
      global_vertices_to_add_.clear();
      global_vertices_to_update_.clear();
      extendVertexMapWithEdges();
    }

    parseEdgesToAdd();

    mutation_t mutation;
    mutation.vertices_to_add.swap(global_vertices_to_add_);
    mutation.vertices_to_remove.swap(global_vertices_to_remove_);
    mutation.vertices_to_update.swap(global_vertices_to_update_);
    mutation.edges_to_add.swap(parsed_edges_to_add_);
    mutation.edges_to_remove.swap(got_edges_to_remove_);
    mutation.edges_to_update.swap(got_edges_to_update_);

    fragment_->Mutate(mutation);

    return fragment_;
  }

  void Start() {
    edges_to_add_.resize(comm_spec_.fnum());
    edges_to_remove_.resize(comm_spec_.fnum());
    edges_to_update_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_add_[fid].Init(comm_spec_.comm(), ea_tag);
      edges_to_add_[fid].SetDestination(worker_id, fid);
      edges_to_remove_[fid].Init(comm_spec_.comm(), er_tag);
      edges_to_remove_[fid].SetDestination(worker_id, fid);
      edges_to_update_[fid].Init(comm_spec_.comm(), eu_tag);
      edges_to_update_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_add_[fid].DisableComm();
        edges_to_remove_[fid].DisableComm();
        edges_to_update_[fid].DisableComm();
      }
    }

    recv_thread_ = std::thread(&BasicFragmentMutator::recvThreadRoutine, this);
  }

  template <typename Q = vdata_t>
  typename std::enable_if<std::is_same<Q, EmptyType>::value>::type AddVertex(
      const oid_t& id, const vdata_t& data) {}

  template <typename Q = vdata_t>
  typename std::enable_if<!std::is_same<Q, EmptyType>::value>::type AddVertex(
      const oid_t& id, const vdata_t& data) {
    local_added_vertices_id_.emplace_back(id);
    local_vertices_to_add_.emplace_back(0, data);
  }

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t src_fid = partitioner.GetPartitionId(src);
    fid_t dst_fid = partitioner.GetPartitionId(dst);
    edges_to_add_[src_fid].Emplace(src, dst, data);
    if (src_fid != dst_fid) {
      edges_to_add_[dst_fid].Emplace(src, dst, data);
    }
  }

  void RemoveVertex(const oid_t& id) {
    vid_t gid;
    if (vm_ptr_->GetGid(id, gid)) {
      local_vertices_to_remove_.push_back(gid);
    }
  }

  void RemoveVertexGidList(std::vector<vid_t>&& gid_list) {
    if (local_vertices_to_remove_.empty()) {
      local_vertices_to_remove_ = std::move(gid_list);
    } else {
      local_vertices_to_remove_.reserve(local_vertices_to_remove_.size() +
                                        gid_list.size());
      for (auto id : gid_list) {
        local_vertices_to_remove_.push_back(id);
      }
    }
  }

  void RemoveEdge(const oid_t& src, const oid_t& dst) {
    vid_t src_gid, dst_gid;
    if (vm_ptr_->GetGid(src, src_gid) && vm_ptr_->GetGid(dst, dst_gid)) {
      if (load_strategy == LoadStrategy::kOnlyOut) {
        fid_t src_fid = vm_ptr_->GetFidFromGid(src_gid);
        edges_to_remove_[src_fid].Emplace(src_gid, dst_gid);
      } else if (load_strategy == LoadStrategy::kOnlyIn) {
        fid_t dst_fid = vm_ptr_->GetFidFromGid(dst_gid);
        edges_to_remove_[dst_fid].Emplace(src_gid, dst_gid);
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        fid_t src_fid = vm_ptr_->GetFidFromGid(src_gid);
        fid_t dst_fid = vm_ptr_->GetFidFromGid(dst_gid);
        edges_to_remove_[src_fid].Emplace(src_gid, dst_gid);
        if (src_fid != dst_fid) {
          edges_to_remove_[dst_fid].Emplace(src_gid, dst_gid);
        }
      } else {
        LOG(FATAL) << "invalid load_strategy";
      }
    }
  }

  void RemoveEdgeGid(const vid_t& src_gid, const vid_t& dst_gid) {
    fid_t src_fid = vm_ptr_->GetFidFromGid(src_gid);
    fid_t dst_fid = vm_ptr_->GetFidFromGid(dst_gid);
    if (load_strategy == LoadStrategy::kOnlyOut) {
      edges_to_remove_[src_fid].Emplace(src_gid, dst_gid);
    } else if (load_strategy == LoadStrategy::kOnlyIn) {
      edges_to_remove_[dst_fid].Emplace(src_gid, dst_gid);
    } else if (load_strategy == LoadStrategy::kBothOutIn) {
      edges_to_remove_[src_fid].Emplace(src_gid, dst_gid);
      if (src_fid != dst_fid) {
        edges_to_remove_[dst_fid].Emplace(src_gid, dst_gid);
      }
    } else {
      LOG(FATAL) << "invalid load_strategy";
    }
  }

  template <typename Q = vdata_t>
  typename std::enable_if<std::is_same<Q, EmptyType>::value>::type UpdateVertex(
      const oid_t& id, const vdata_t& data) {}

  template <typename Q = vdata_t>
  typename std::enable_if<std::is_same<Q, EmptyType>::value>::type
  UpdateVertexGidList(
      std::vector<internal::Vertex<vid_t, vdata_t>>&& vertex_list) {}

  template <typename Q = vdata_t>
  typename std::enable_if<!std::is_same<Q, EmptyType>::value>::type
  UpdateVertex(const oid_t& id, const vdata_t& data) {
    vid_t gid;
    if (vm_ptr_->GetGid(id, gid)) {
      local_vertices_to_update_.emplace_back(gid, data);
    }
  }

  template <typename Q = vdata_t>
  typename std::enable_if<!std::is_same<Q, EmptyType>::value>::type
  UpdateVertexGidList(
      std::vector<internal::Vertex<vid_t, vdata_t>>&& vertex_list) {
    if (local_vertices_to_update_.empty()) {
      local_vertices_to_update_ = std::move(vertex_list);
    } else {
      local_vertices_to_update_.reserve(local_vertices_to_update_.size() +
                                        vertex_list.size());
      for (auto& v : vertex_list) {
        local_vertices_to_update_.emplace_back(std::move(v));
      }
    }
  }

  void UpdateEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    vid_t src_gid, dst_gid;
    if (vm_ptr_->GetGid(src, src_gid) && vm_ptr_->GetGid(dst, dst_gid)) {
      if (load_strategy == LoadStrategy::kOnlyOut) {
        fid_t src_fid = vm_ptr_->GetFidFromGid(src_gid);
        edges_to_update_[src_fid].Emplace(src_gid, dst_gid, data);
      } else if (load_strategy == LoadStrategy::kOnlyIn) {
        fid_t dst_fid = vm_ptr_->GetFidFromGid(dst_gid);
        edges_to_update_[dst_fid].Emplace(src_gid, dst_gid, data);
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        fid_t src_fid = vm_ptr_->GetFidFromGid(src_gid);
        fid_t dst_fid = vm_ptr_->GetFidFromGid(dst_gid);
        edges_to_update_[src_fid].Emplace(src_gid, dst_gid, data);
        if (src_fid != dst_fid) {
          edges_to_update_[dst_fid].Emplace(src_gid, dst_gid, data);
        }
      } else {
        LOG(FATAL) << "invalid load_strategy";
      }
    }
  }

  void UpdateEdgeGid(const vid_t& src_gid, const vid_t& dst_gid,
                     const edata_t& data) {
    fid_t src_fid = vm_ptr_->GetFidFromGid(src_gid);
    fid_t dst_fid = vm_ptr_->GetFidFromGid(dst_gid);
    if (load_strategy == LoadStrategy::kOnlyOut) {
      edges_to_update_[src_fid].Emplace(src_gid, dst_gid, data);
    } else if (load_strategy == LoadStrategy::kOnlyIn) {
      edges_to_update_[dst_fid].Emplace(src_gid, dst_gid, data);
    } else if (load_strategy == LoadStrategy::kBothOutIn) {
      edges_to_update_[src_fid].Emplace(src_gid, dst_gid, data);
      if (src_fid != dst_fid) {
        edges_to_update_[dst_fid].Emplace(src_gid, dst_gid, data);
      }
    } else {
      LOG(FATAL) << "invalid load_strategy";
    }
  }

 private:
  void recvThreadRoutine() {
    if (comm_spec_.fnum() == 1) {
      return;
    }
    ShuffleIn<oid_t, oid_t, edata_t> edges_to_add_in;
    edges_to_add_in.Init(comm_spec_.fnum(), comm_spec_.comm(), ea_tag);
    ShuffleIn<vid_t, vid_t> edges_to_remove_in;
    edges_to_remove_in.Init(comm_spec_.fnum(), comm_spec_.comm(), er_tag);
    ShuffleIn<vid_t, vid_t, edata_t> edges_to_update_in;
    edges_to_update_in.Init(comm_spec_.fnum(), comm_spec_.comm(), eu_tag);

    int remaining_channel = 3;
    while (remaining_channel != 0) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_spec_.comm(), &status);
      if (status.MPI_TAG == ea_tag) {
        if (edges_to_add_in.RecvFrom(status.MPI_SOURCE)) {
          std::vector<oid_t>& src_list = get_buffer<0>(edges_to_add_in);
          std::vector<oid_t>& dst_list = get_buffer<1>(edges_to_add_in);
          std::vector<edata_t>& data_list = get_buffer<2>(edges_to_add_in);
          size_t num = src_list.size();
          for (size_t k = 0; k < num; ++k) {
            got_edges_to_add_.emplace_back(std::move(src_list[k]),
                                           std::move(dst_list[k]),
                                           std::move(data_list[k]));
          }
          edges_to_add_in.Clear();
        }
        if (edges_to_add_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == er_tag) {
        if (edges_to_remove_in.RecvFrom(status.MPI_SOURCE)) {
          std::vector<vid_t>& src_list = get_buffer<0>(edges_to_remove_in);
          std::vector<vid_t>& dst_list = get_buffer<1>(edges_to_remove_in);
          size_t num = src_list.size();
          for (size_t k = 0; k < num; ++k) {
            got_edges_to_remove_.emplace_back(src_list[k], dst_list[k]);
          }
          edges_to_remove_in.Clear();
        }
        if (edges_to_remove_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == eu_tag) {
        if (edges_to_update_in.RecvFrom(status.MPI_SOURCE)) {
          std::vector<vid_t>& src_list = get_buffer<0>(edges_to_update_in);
          std::vector<vid_t>& dst_list = get_buffer<1>(edges_to_update_in);
          std::vector<edata_t>& data_list = get_buffer<2>(edges_to_update_in);
          size_t num = src_list.size();
          for (size_t k = 0; k < num; ++k) {
            got_edges_to_update_.emplace_back(src_list[k], dst_list[k],
                                              std::move(data_list[k]));
          }
          edges_to_update_in.Clear();
        }
        if (edges_to_update_in.Finished()) {
          --remaining_channel;
        }
      } else {
        LOG(FATAL) << "unexpected tag: " << status.MPI_TAG;
      }
    }
  }

  void extendVertexMap(const std::vector<oid_t>& oid_list) {
    for (auto& id : oid_list) {
      vm_ptr_->AddVertex(id);
    }
  }

  void extendVertexMapWithEdges() {
    fid_t fid = comm_spec_.fid();
    size_t ivnum_before = vm_ptr_->GetInnerVertexSize(fid);
    auto builder = vm_ptr_->GetLocalBuilder();
    auto& partitioner = vm_ptr_->GetPartitioner();
    for (auto& e : got_edges_to_add_) {
      fid_t src_fid = partitioner.GetPartitionId(e.src);
      fid_t dst_fid = partitioner.GetPartitionId(e.dst);
      if (src_fid == fid) {
        builder.add_vertex(e.src);
      }
      if (dst_fid == fid) {
        builder.add_vertex(e.dst);
      }
    }
    size_t ivnum_after = vm_ptr_->GetInnerVertexSize(fid);
    VLOG(1) << "[frag-" << fid << "] added " << ivnum_after - ivnum_before
            << " vertices";
    builder.finish(*vm_ptr_);
  }

  void processToSelfMessages() {
    {
      std::vector<oid_t>& src_list =
          get_buffer<0>(edges_to_add_[comm_spec_.fid()]);
      std::vector<oid_t>& dst_list =
          get_buffer<1>(edges_to_add_[comm_spec_.fid()]);
      std::vector<edata_t>& data_list =
          get_buffer<2>(edges_to_add_[comm_spec_.fid()]);
      size_t num = src_list.size();
      for (size_t k = 0; k < num; ++k) {
        got_edges_to_add_.emplace_back(std::move(src_list[k]),
                                       std::move(dst_list[k]),
                                       std::move(data_list[k]));
      }
      edges_to_add_[comm_spec_.fid()].Clear();
    }
    {
      std::vector<vid_t>& src_list =
          get_buffer<0>(edges_to_remove_[comm_spec_.fid()]);
      std::vector<vid_t>& dst_list =
          get_buffer<1>(edges_to_remove_[comm_spec_.fid()]);
      size_t num = src_list.size();
      for (size_t k = 0; k < num; ++k) {
        got_edges_to_remove_.emplace_back(src_list[k], dst_list[k]);
      }
      edges_to_remove_[comm_spec_.fid()].Clear();
    }
    {
      std::vector<vid_t>& src_list =
          get_buffer<0>(edges_to_update_[comm_spec_.fid()]);
      std::vector<vid_t>& dst_list =
          get_buffer<1>(edges_to_update_[comm_spec_.fid()]);
      std::vector<edata_t>& data_list =
          get_buffer<2>(edges_to_update_[comm_spec_.fid()]);
      size_t num = src_list.size();
      for (size_t k = 0; k < num; ++k) {
        got_edges_to_update_.emplace_back(src_list[k], dst_list[k],
                                          std::move(data_list[k]));
      }
      edges_to_update_[comm_spec_.fid()].Clear();
    }
  }

  void parseEdgesToAdd() {
    size_t edge_num = got_edges_to_add_.size();
    parsed_edges_to_add_.resize(edge_num);
    for (size_t i = 0; i < edge_num; ++i) {
      auto& ei = got_edges_to_add_[i];
      auto& eo = parsed_edges_to_add_[i];
      if (!(vm_ptr_->GetGid(ei.src, eo.src) &&
            vm_ptr_->GetGid(ei.dst, eo.dst))) {
        VLOG(10) << "edge parse failed: " << ei.src << " " << ei.dst << " "
                 << ei.edata;
        eo.src = std::numeric_limits<vid_t>::max();
      } else {
        eo.edata = std::move(ei.edata);
      }
    }
  }

  CommSpec comm_spec_;
  std::shared_ptr<fragment_t> fragment_;
  std::shared_ptr<vertex_map_t> vm_ptr_;

  std::thread recv_thread_;

  std::vector<oid_t> local_added_vertices_id_;
  std::vector<internal::Vertex<vid_t, vdata_t>> local_vertices_to_add_;
  std::vector<internal::Vertex<vid_t, vdata_t>> global_vertices_to_add_;
  std::vector<internal::Vertex<vid_t, vdata_t>> local_vertices_to_update_;
  std::vector<internal::Vertex<vid_t, vdata_t>> global_vertices_to_update_;
  std::vector<vid_t> local_vertices_to_remove_;
  std::vector<vid_t> global_vertices_to_remove_;

  std::vector<ShuffleOut<oid_t, oid_t, edata_t>> edges_to_add_;
  static constexpr int ea_tag = 1;
  std::vector<ShuffleOut<vid_t, vid_t>> edges_to_remove_;
  static constexpr int er_tag = 2;
  std::vector<ShuffleOut<vid_t, vid_t, edata_t>> edges_to_update_;
  static constexpr int eu_tag = 3;

  std::vector<Edge<oid_t, edata_t>> got_edges_to_add_;
  std::vector<Edge<vid_t, edata_t>> parsed_edges_to_add_;
  std::vector<std::pair<vid_t, vid_t>> got_edges_to_remove_;
  std::vector<Edge<vid_t, edata_t>> got_edges_to_update_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_MUTATOR_H_
