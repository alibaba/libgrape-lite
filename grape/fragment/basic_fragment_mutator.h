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
  using internal_oid_t = typename InternalOID<oid_t>::type;
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

  void AddVerticesToRemove(const std::vector<vid_t>& id_vec) {
    if (parsed_vertices_to_remove_.empty()) {
      parsed_vertices_to_remove_ = id_vec;
    } else {
      for (auto v : id_vec) {
        parsed_vertices_to_remove_.emplace_back(v);
      }
    }
  }

  void AddVerticesToRemove(std::vector<vid_t>&& id_vec) {
    if (parsed_vertices_to_remove_.empty()) {
      parsed_vertices_to_remove_ = std::move(id_vec);
    } else {
      for (auto v : id_vec) {
        parsed_vertices_to_remove_.emplace_back(v);
      }
    }
  }

  void AddVerticesToUpdate(
      const std::vector<internal::Vertex<vid_t, vdata_t>>& v_vec) {
    if (parsed_vertices_to_update_.empty()) {
      parsed_vertices_to_update_ = v_vec;
    } else {
      for (auto& v : v_vec) {
        parsed_vertices_to_update_.emplace_back(v);
      }
    }
  }

  void AddVerticesToUpdate(
      std::vector<internal::Vertex<vid_t, vdata_t>>&& v_vec) {
    if (parsed_vertices_to_update_.empty()) {
      parsed_vertices_to_update_ = std::move(v_vec);
    } else {
      for (auto& v : v_vec) {
        parsed_vertices_to_update_.emplace_back(v);
      }
    }
  }

  std::shared_ptr<fragment_t> MutateFragment() {
    for (auto& shuf : vertices_to_add_) {
      shuf.Flush();
    }
    for (auto& shuf : vertices_to_remove_) {
      shuf.Flush();
    }
    for (auto& shuf : vertices_to_update_) {
      shuf.Flush();
    }
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
    got_vertices_to_add_.emplace_back(
        std::move(vertices_to_add_[comm_spec_.fid()].buffers()));
    got_vertices_to_remove_.emplace_back(
        std::move(vertices_to_remove_[comm_spec_.fid()].buffers()));
    got_vertices_to_update_.emplace_back(
        std::move(vertices_to_update_[comm_spec_.fid()].buffers()));
    got_edges_to_add_.emplace_back(
        std::move(edges_to_add_[comm_spec_.fid()].buffers()));
    got_edges_to_remove_.emplace_back(
        std::move(edges_to_remove_[comm_spec_.fid()].buffers()));
    got_edges_to_update_.emplace_back(
        std::move(edges_to_update_[comm_spec_.fid()].buffers()));

    if (!std::is_same<edata_t, grape::EmptyType>::value) {
      for (auto& buffers : got_edges_to_update_) {
        foreach_rval(buffers, [this](internal_oid_t&& src, internal_oid_t&& dst,
                                     edata_t&& data) {
          vid_t src_gid, dst_gid;
          if (vm_ptr_->_GetGid(src, src_gid) &&
              vm_ptr_->_GetGid(dst, dst_gid)) {
            mutation_.edges_to_update.emplace_back(src_gid, dst_gid,
                                                   std::move(data));
          }
        });
      }
    }
    got_edges_to_update_.clear();

    for (auto& buffers : got_edges_to_remove_) {
      foreach(buffers, [this](const internal_oid_t& src,
                              const internal_oid_t& dst) {
        vid_t src_gid, dst_gid;
        if (vm_ptr_->_GetGid(src, src_gid) && vm_ptr_->_GetGid(dst, dst_gid)) {
          mutation_.edges_to_remove.emplace_back(src_gid, dst_gid);
        }
      });
    }
    got_edges_to_remove_.clear();

    for (auto& buffers : got_vertices_to_remove_) {
      foreach(buffers, [this](const internal_oid_t& id) {
        vid_t gid;
        if (vm_ptr_->_GetGid(id, gid)) {
          parsed_vertices_to_remove_.emplace_back(gid);
        }
      });
    }
    got_vertices_to_remove_.clear();

    if (!std::is_same<vdata_t, grape::EmptyType>::value) {
      for (auto& buffers : got_vertices_to_update_) {
        foreach_rval(buffers, [this](internal_oid_t&& id, vdata_t&& data) {
          vid_t gid;
          if (vm_ptr_->_GetGid(id, gid)) {
            parsed_vertices_to_update_.emplace_back(gid, std::move(data));
          }
        });
      }
    }
    got_vertices_to_update_.clear();

    auto builder = vm_ptr_->GetLocalBuilder();
    for (auto& buffers : got_vertices_to_add_) {
      foreach_rval(buffers,
                   [this, &builder](internal_oid_t&& id, vdata_t&& data) {
                     vid_t gid;
                     builder.add_local_vertex(id, gid);
                     parsed_vertices_to_add_.emplace_back(gid, std::move(data));
                   });
    }
    got_vertices_to_add_.clear();

    for (auto& buffers : got_edges_to_add_) {
      foreach_helper(
          buffers,
          [&builder](const internal_oid_t& src, const internal_oid_t& dst) {
            builder.add_vertex(src);
            builder.add_vertex(dst);
          },
          make_index_sequence<2>{});
    }
    builder.finish(*vm_ptr_);

    for (auto& buffers : got_edges_to_add_) {
      foreach_rval(buffers, [this](internal_oid_t&& src, internal_oid_t&& dst,
                                   edata_t&& data) {
        vid_t src_gid, dst_gid;
        if (vm_ptr_->_GetGid(src, src_gid) && vm_ptr_->_GetGid(dst, dst_gid)) {
          mutation_.edges_to_add.emplace_back(src_gid, dst_gid,
                                              std::move(data));
        }
      });
    }
    got_edges_to_add_.clear();

    sync_comm::FlatAllGather<vid_t>(parsed_vertices_to_remove_,
                                    mutation_.vertices_to_remove,
                                    comm_spec_.comm());
    sync_comm::FlatAllGather<internal::Vertex<vid_t, vdata_t>>(
        parsed_vertices_to_update_, mutation_.vertices_to_update,
        comm_spec_.comm());
    sync_comm::FlatAllGather<internal::Vertex<vid_t, vdata_t>>(
        parsed_vertices_to_add_, mutation_.vertices_to_add, comm_spec_.comm());

    fragment_->Mutate(mutation_);

    return fragment_;
  }

  void Start() {
    vertices_to_add_.resize(comm_spec_.fnum());
    vertices_to_remove_.resize(comm_spec_.fnum());
    vertices_to_update_.resize(comm_spec_.fnum());
    edges_to_add_.resize(comm_spec_.fnum());
    edges_to_remove_.resize(comm_spec_.fnum());
    edges_to_update_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      vertices_to_add_[fid].Init(comm_spec_.comm(), va_tag);
      vertices_to_add_[fid].SetDestination(worker_id, fid);
      vertices_to_remove_[fid].Init(comm_spec_.comm(), vr_tag);
      vertices_to_remove_[fid].SetDestination(worker_id, fid);
      vertices_to_update_[fid].Init(comm_spec_.comm(), vu_tag);
      vertices_to_update_[fid].SetDestination(worker_id, fid);
      edges_to_add_[fid].Init(comm_spec_.comm(), ea_tag);
      edges_to_add_[fid].SetDestination(worker_id, fid);
      edges_to_remove_[fid].Init(comm_spec_.comm(), er_tag);
      edges_to_remove_[fid].SetDestination(worker_id, fid);
      edges_to_update_[fid].Init(comm_spec_.comm(), eu_tag);
      edges_to_update_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        vertices_to_add_[fid].DisableComm();
        vertices_to_remove_[fid].DisableComm();
        vertices_to_update_[fid].DisableComm();
        edges_to_add_[fid].DisableComm();
        edges_to_remove_[fid].DisableComm();
        edges_to_update_[fid].DisableComm();
      }
    }

    recv_thread_ = std::thread(&BasicFragmentMutator::recvThreadRoutine, this);
  }

  void AddVertex(const internal_oid_t& id, const vdata_t& data) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t fid = partitioner.GetPartitionId(id);
    vertices_to_add_[fid].Emplace(id, data);
  }

  void AddVertices(
      std::vector<typename ShuffleBuffer<internal_oid_t>::type>&& id_lists,
      std::vector<typename ShuffleBuffer<vdata_t>::type>&& data_lists) {
    CHECK_EQ(id_lists.size(), vertices_to_add_.size());
    CHECK_EQ(data_lists.size(), vertices_to_add_.size());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      vertices_to_add_[i].AppendBuffers(std::move(id_lists[i]),
                                        std::move(data_lists[i]));
    }
  }

  void AddEdge(const internal_oid_t& src, const internal_oid_t& dst,
               const edata_t& data) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t src_fid = partitioner.GetPartitionId(src);
    fid_t dst_fid = partitioner.GetPartitionId(dst);
    edges_to_add_[src_fid].Emplace(src, dst, data);
    if (src_fid != dst_fid) {
      edges_to_add_[dst_fid].Emplace(src, dst, data);
    }
  }

  void AddEdges(
      std::vector<typename ShuffleBuffer<internal_oid_t>::type>&& src_lists,
      std::vector<typename ShuffleBuffer<internal_oid_t>::type>&& dst_lists,
      std::vector<typename ShuffleBuffer<edata_t>::type>&& data_lists) {
    CHECK_EQ(src_lists.size(), edges_to_add_.size());
    CHECK_EQ(dst_lists.size(), edges_to_add_.size());
    CHECK_EQ(data_lists.size(), edges_to_add_.size());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      edges_to_add_[i].AppendBuffers(std::move(src_lists[i]),
                                     std::move(dst_lists[i]),
                                     std::move(data_lists[i]));
    }
  }

  void RemoveVertex(const oid_t& id) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t fid = partitioner.GetPartitionId(id);
    vertices_to_remove_[fid].Emplace(id);
  }

  void RemoveVertices(
      std::vector<typename ShuffleBuffer<oid_t>::type>&& id_lists) {
    CHECK_EQ(id_lists.size(), vertices_to_remove_.size());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      vertices_to_remove_[i].AppendBuffers(std::move(id_lists[i]));
    }
  }

  void RemoveEdge(const oid_t& src, const oid_t& dst) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t src_fid = partitioner.GetPartitionId(src);
    fid_t dst_fid = partitioner.GetPartitionId(dst);
    edges_to_remove_[src_fid].Emplace(src, dst);
    if (src_fid != dst_fid) {
      edges_to_remove_[dst_fid].Emplace(src, dst);
    }
  }

  void RemoveEdges(
      std::vector<typename ShuffleBuffer<oid_t>::type>&& src_lists,
      std::vector<typename ShuffleBuffer<oid_t>::type>&& dst_lists) {
    CHECK_EQ(src_lists.size(), edges_to_add_.size());
    CHECK_EQ(dst_lists.size(), edges_to_add_.size());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      edges_to_remove_[i].AppendBuffers(std::move(src_lists[i]),
                                        std::move(dst_lists[i]));
    }
  }

  template <typename Q = vdata_t>
  typename std::enable_if<std::is_same<Q, EmptyType>::value>::type UpdateVertex(
      const oid_t& id, const vdata_t& data) {}

  template <typename Q = vdata_t>
  typename std::enable_if<!std::is_same<Q, EmptyType>::value>::type
  UpdateVertex(const oid_t& id, const vdata_t& data) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t fid = partitioner.GetPartitionId(id);
    vertices_to_update_[fid].Emplace(id, data);
  }

  template <typename Q = vdata_t>
  typename std::enable_if<std::is_same<Q, EmptyType>::value>::type
  UpdateVertices(
      std::vector<typename ShuffleBuffer<oid_t>::type>&& id_lists,
      std::vector<typename ShuffleBuffer<vdata_t>::type>&& data_lists) {}

  template <typename Q = vdata_t>
  typename std::enable_if<!std::is_same<Q, EmptyType>::value>::type
  UpdateVertices(
      std::vector<typename ShuffleBuffer<oid_t>::type>&& id_lists,
      std::vector<typename ShuffleBuffer<vdata_t>::type>&& data_lists) {
    CHECK_EQ(id_lists.size(), vertices_to_update_.size());
    CHECK_EQ(data_lists.size(), vertices_to_update_.size());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      vertices_to_update_[i].AppendBuffers(std::move(id_lists[i]),
                                           std::move(data_lists[i]));
    }
  }

  void UpdateEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t src_fid = partitioner.GetPartitionId(src);
    fid_t dst_fid = partitioner.GetPartitionId(dst);
    edges_to_update_[src_fid].Emplace(src, dst, data);
    if (src_fid != dst_fid) {
      edges_to_update_[dst_fid].Emplace(src, dst, data);
    }
  }

  void UpdateEdges(
      std::vector<typename ShuffleBuffer<oid_t>::type>&& src_lists,
      std::vector<typename ShuffleBuffer<oid_t>::type>&& dst_lists,
      std::vector<typename ShuffleBuffer<edata_t>::type>&& data_lists) {
    CHECK_EQ(src_lists.size(), edges_to_update_.size());
    CHECK_EQ(dst_lists.size(), edges_to_update_.size());
    CHECK_EQ(data_lists.size(), edges_to_update_.size());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      edges_to_update_[i].AppendBuffers(std::move(src_lists[i]),
                                        std::move(dst_lists[i]),
                                        std::move(data_lists[i]));
    }
  }

 private:
  void recvThreadRoutine() {
    if (comm_spec_.fnum() == 1) {
      return;
    }
    ShuffleIn<internal_oid_t> vertices_to_remove_in;
    vertices_to_remove_in.Init(comm_spec_.fnum(), comm_spec_.comm(), vr_tag);
    ShuffleIn<internal_oid_t, vdata_t> vertices_to_update_in;
    vertices_to_update_in.Init(comm_spec_.fnum(), comm_spec_.comm(), vu_tag);
    ShuffleIn<internal_oid_t, vdata_t> vertices_to_add_in;
    vertices_to_add_in.Init(comm_spec_.fnum(), comm_spec_.comm(), va_tag);
    ShuffleIn<internal_oid_t, internal_oid_t, edata_t> edges_to_add_in;
    edges_to_add_in.Init(comm_spec_.fnum(), comm_spec_.comm(), ea_tag);
    ShuffleIn<internal_oid_t, internal_oid_t> edges_to_remove_in;
    edges_to_remove_in.Init(comm_spec_.fnum(), comm_spec_.comm(), er_tag);
    ShuffleIn<internal_oid_t, internal_oid_t, edata_t> edges_to_update_in;
    edges_to_update_in.Init(comm_spec_.fnum(), comm_spec_.comm(), eu_tag);

    int remaining_channel = 6;
    while (remaining_channel != 0) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_spec_.comm(), &status);
      if (status.MPI_TAG == va_tag) {
        if (vertices_to_add_in.RecvFrom(status.MPI_SOURCE)) {
          got_vertices_to_add_.emplace_back(
              std::move(vertices_to_add_in.buffers()));
          vertices_to_add_in.Clear();
        }
        if (vertices_to_add_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == vr_tag) {
        if (vertices_to_remove_in.RecvFrom(status.MPI_SOURCE)) {
          got_vertices_to_remove_.emplace_back(
              std::move(vertices_to_remove_in.buffers()));
          vertices_to_remove_in.Clear();
        }
        if (vertices_to_remove_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == vu_tag) {
        if (vertices_to_update_in.RecvFrom(status.MPI_SOURCE)) {
          got_vertices_to_update_.emplace_back(
              std::move(vertices_to_update_in.buffers()));
          vertices_to_update_in.Clear();
        }
        if (vertices_to_update_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == ea_tag) {
        if (edges_to_add_in.RecvFrom(status.MPI_SOURCE)) {
          got_edges_to_add_.emplace_back(std::move(edges_to_add_in.buffers()));
          edges_to_add_in.Clear();
        }
        if (edges_to_add_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == er_tag) {
        if (edges_to_remove_in.RecvFrom(status.MPI_SOURCE)) {
          got_edges_to_remove_.emplace_back(
              std::move(edges_to_remove_in.buffers()));
          edges_to_remove_in.Clear();
        }
        if (edges_to_remove_in.Finished()) {
          --remaining_channel;
        }
      } else if (status.MPI_TAG == eu_tag) {
        if (edges_to_update_in.RecvFrom(status.MPI_SOURCE)) {
          got_edges_to_update_.emplace_back(
              std::move(edges_to_update_in.buffers()));
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

  CommSpec comm_spec_;
  std::shared_ptr<fragment_t> fragment_;
  std::shared_ptr<vertex_map_t> vm_ptr_;

  std::thread recv_thread_;

  std::vector<vid_t> parsed_vertices_to_remove_;
  std::vector<internal::Vertex<vid_t, vdata_t>> parsed_vertices_to_update_;
  std::vector<internal::Vertex<vid_t, vdata_t>> parsed_vertices_to_add_;

  std::vector<ShuffleOut<internal_oid_t>> vertices_to_remove_;
  std::vector<ShuffleBufferTuple<internal_oid_t>> got_vertices_to_remove_;
  static constexpr int vr_tag = 1;
  std::vector<ShuffleOut<internal_oid_t, vdata_t>> vertices_to_add_;
  std::vector<ShuffleBufferTuple<internal_oid_t, vdata_t>> got_vertices_to_add_;
  static constexpr int va_tag = 2;
  std::vector<ShuffleOut<internal_oid_t, vdata_t>> vertices_to_update_;
  std::vector<ShuffleBufferTuple<internal_oid_t, vdata_t>>
      got_vertices_to_update_;
  static constexpr int vu_tag = 3;

  std::vector<ShuffleOut<internal_oid_t, internal_oid_t>> edges_to_remove_;
  std::vector<ShuffleBufferTuple<internal_oid_t, internal_oid_t>>
      got_edges_to_remove_;
  static constexpr int er_tag = 4;
  std::vector<ShuffleOut<internal_oid_t, internal_oid_t, edata_t>>
      edges_to_update_;
  std::vector<ShuffleBufferTuple<internal_oid_t, internal_oid_t, edata_t>>
      got_edges_to_update_;
  static constexpr int ea_tag = 5;
  std::vector<ShuffleOut<internal_oid_t, internal_oid_t, edata_t>>
      edges_to_add_;
  std::vector<ShuffleBufferTuple<internal_oid_t, internal_oid_t, edata_t>>
      got_edges_to_add_;
  static constexpr int eu_tag = 6;

  mutation_t mutation_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_MUTATOR_H_
