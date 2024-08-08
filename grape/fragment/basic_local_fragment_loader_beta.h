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

#ifndef GRAPE_FRAGMENT_BASIC_LOCAL_FRAGMENT_LOADER_BETA_H_
#define GRAPE_FRAGMENT_BASIC_LOCAL_FRAGMENT_LOADER_BETA_H_

namespace grape {

template <typename FRAG_T>
class BasicLocalFragmentLoaderBeta : public BasicFragmentLoaderBase<FRAG_T> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using internal_oid_t = typename InternalOID<oid_t>::type;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

 public:
  explicit BasicLocalFragmentLoaderBeta(const CommSpec& comm_spec,
                                        const LoadGraphSpec& spec)
      : BasicFragmentLoaderBase<FRAG_T>(comm_spec, spec) {
    if (spec_.global_vertex_map) {
      LOG(ERROR)
          << "Local vertex map is required in BasicLocalFragmentLoaderBeta";
      spec_.global_vertex_map = false;
    }
    if (!spec_.mutable_vertex_map) {
      LOG(ERROR)
          << "Mutable vertex map is required in BasicLocalFragmentLoaderBeta";
      spec_.mutable_vertex_map = true;
    }
    if (spec_.rebalance) {
      LOG(ERROR)
          << "Rebalance is not supported in BasicLocalFragmentLoaderBeta";
      spec_.rebalance = false;
    }
    if (spec_.partitioner_type != PartitionerType::kHashPartitioner) {
      LOG(ERROR) << "Only hash partitioner is supported in "
                    "BasicLocalFragmentLoaderBeta";
    }
    partitioner_ = std::unique_ptr<HashPartitionerBeta<oid_t>>(
        new HashPartitionerBeta<oid_t>(comm_spec_.fnum()));

    vertices_to_frag_.resize(comm_spec_.fnum());
    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      vertices_to_frag_[fid].Init(comm_spec_.comm(), vertex_tag, 4096000);
      vertices_to_frag_[fid].SetDestination(worker_id, fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag, 4096000);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        vertices_to_frag_[fid].DisableComm();
        edges_to_frag_[fid].DisableComm();
      }
    }

    vertex_recv_thread_ =
        std::thread(&BasicLocalFragmentLoaderBeta::vertexRecvRoutine, this);
    vertex_recv_thread_running_ = true;
  }

  ~BasicLocalFragmentLoaderBeta() {
    if (!vertex_recv_thread_running_) {
      for (auto& va : vertices_to_frag_) {
        va.Flush();
      }
      vertex_recv_thread_.join();
    }
    if (!edge_recv_thread_running_) {
      for (auto& ea : edges_to_frag_) {
        ea.Flush();
      }
      edge_recv_thread_.join();
    }
  }

  void AddVertex(const oid_t& id, const vdata_t& data) override {
    internal_oid_t internal_id(id);
    fid_t fid = partitioner_->GetPartitionId(internal_id);
    if (fid == comm_spec_.fnum()) {
      LOG(ERROR) << "Unknown partition id for vertex " << id;
    } else {
      vertices_to_frag_[fid].Emplace(internal_id, data);
    }
  }

  void ConstructVertices() override {
    for (auto& va : vertices_to_frag_) {
      va.Flush();
    }
    vertex_recv_thread_.join();
    vertex_recv_thread_running_ = false;

    got_vertices_.emplace_back(
        std::move(vertices_to_frag_[comm_spec_.fid()].buffers()));
    vertices_to_frag_[comm_spec_.fid()].Clear();

    edge_recv_thread_ =
        std::thread(&BasicLocalFragmentLoaderBeta::edgeRecvRoutine, this);
    edge_recv_thread_running_ = true;
  }

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
    edge_recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());
    got_edges_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].buffers()));
    edges_to_frag_[comm_spec_.fid()].Clear();

    std::unique_ptr<VertexMap<oid_t, vid_t>> vm_ptr(
        new VertexMap<oid_t, vid_t>());
    {
      VertexMapBuilder<oid_t, vid_t> builder(
          comm_spec_.fid(), comm_spec_.fnum(), std::move(partitioner_),
          spec_.global_vertex_map, !spec_.mutable_vertex_map);
      for (auto& buffers : got_vertices_) {
        foreach_helper(
            buffers,
            [&builder](const internal_oid_t& id) { builder.add_vertex(id); },
            make_index_sequence<1>{});
      }
      for (auto& buffers : got_edges_) {
        foreach_helper(
            buffers,
            [&builder](const internal_oid_t& src, const internal_oid_t& dst) {
              builder.add_vertex(src);
              builder.add_vertex(dst);
            },
            make_index_sequence<2>{});
      }
      builder.finish(comm_spec_, *vm_ptr);
    }

    std::vector<internal::Vertex<vid_t, vdata_t>> processed_vertices;
    for (auto& buffers : got_vertices_) {
      foreach_rval(buffers, [this, &vm_ptr, &processed_vertices](
                                internal_oid_t&& id, vdata_t&& data) {
        vid_t gid;
        CHECK(vm_ptr->GetGid(oid_t(id), gid));
        processed_vertices.emplace_back(gid, std::move(data));
      });
    }

    std::vector<Edge<vid_t, edata_t>> processed_edges;
    for (auto& buffers : got_edges_) {
      foreach_rval(buffers, [this, &vm_ptr, &processed_edges](
                                internal_oid_t&& src, internal_oid_t&& dst,
                                edata_t&& data) {
        vid_t src_gid, dst_gid;
        CHECK(vm_ptr->GetGid(oid_t(src), src_gid));
        CHECK(vm_ptr->GetGid(oid_t(dst), dst_gid));
        processed_edges.emplace_back(src_gid, dst_gid, std::move(data));
      });
    }

    fragment = std::shared_ptr<fragment_t>(new fragment_t());
    fragment->Init(comm_spec_.fid(), spec_.directed, std::move(vm_ptr),
                   processed_vertices, processed_edges);

    if (!std::is_same<vdata_t, EmptyType>::value) {
      this->InitOuterVertexData(fragment);
    }
  }

 private:
  void vertexRecvRoutine() {
    ShuffleIn<internal_oid_t, vdata_t> data_in;
    data_in.Init(comm_spec_.fnum(), comm_spec_.comm(), vertex_tag);
    fid_t dst_fid;
    int src_worker_id;
    while (!data_in.Finished()) {
      src_worker_id = data_in.Recv(dst_fid);
      if (src_worker_id == -1) {
        break;
      }
      got_vertices_.emplace_back(std::move(data_in.buffers()));
      data_in.Clear();
    }
  }

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

  std::unique_ptr<HashPartitionerBeta<oid_t>> partitioner_;

  std::vector<ShuffleOut<internal_oid_t, vdata_t>> vertices_to_frag_;
  std::vector<ShuffleOut<internal_oid_t, internal_oid_t, edata_t>>
      edges_to_frag_;

  std::thread vertex_recv_thread_;
  bool vertex_recv_thread_running_;
  std::thread edge_recv_thread_;
  bool edge_recv_thread_running_;

  std::vector<ShuffleBufferTuple<internal_oid_t, vdata_t>> got_vertices_;
  std::vector<ShuffleBufferTuple<internal_oid_t, internal_oid_t, edata_t>>
      got_edges_;

  using BasicFragmentLoaderBase<FRAG_T>::comm_spec_;
  using BasicFragmentLoaderBase<FRAG_T>::spec_;
  using BasicFragmentLoaderBase<FRAG_T>::id_parser_;

  using BasicFragmentLoaderBase<FRAG_T>::vertex_tag;
  using BasicFragmentLoaderBase<FRAG_T>::edge_tag;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_LOCAL_FRAGMENT_LOADER_BETA_H_
