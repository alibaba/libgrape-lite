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

#ifndef GRAPE_FRAGMENT_BASIC_RB_FRAGMENT_LOADER_BETA_H_
#define GRAPE_FRAGMENT_BASIC_RB_FRAGMENT_LOADER_BETA_H_

#include "grape/fragment/basic_fragment_loader_base.h"

namespace grape {

template <typename FRAG_T>
class BasicRbFragmentLoaderBeta : public BasicFragmentLoaderBase<FRAG_T> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using internal_oid_t = typename InternalOID<oid_t>::type;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

 public:
  explicit BasicRbFragmentLoaderBeta(const CommSpec& comm_spec,
                                     const LoadGraphSpec& spec)
      : BasicFragmentLoaderBase<FRAG_T>(comm_spec, spec) {
    if (!spec_.global_vertex_map) {
      LOG(ERROR) << "Global vertex map is required in BasicFragmentLoaderBeta";
      spec_.global_vertex_map = true;
    }
  }

  ~BasicRbFragmentLoaderBeta() {}

  void AddVertex(const oid_t& id, const vdata_t& data) override {
    vertices_.emplace_back(id);
    vdata_.emplace_back(data);
  }

  void ConstructVertices() override {
    fid_t fid = comm_spec_.fid();
    fid_t fnum = comm_spec_.fnum();
    std::unique_ptr<IPartitioner<oid_t>> partitioner(nullptr);
    if (spec_.partitioner_type == PartitionerType::kHashPartitioner) {
      partitioner = std::unique_ptr<HashPartitionerBeta<oid_t>>(
          new HashPartitionerBeta<oid_t>(fnum));
    } else if (spec_.partitioner_type == PartitionerType::kMapPartitioner) {
      std::vector<oid_t> all_vertices;
      sync_comm::FlatAllGather(vertices_, all_vertices, comm_spec_.comm());
      DistinctSort(all_vertices);

      partitioner = std::unique_ptr<MapPartitioner<oid_t>>(
          new MapPartitioner<oid_t>(fnum, all_vertices));
    } else {
      LOG(FATAL) << "Unsupported partitioner type";
    }

    std::vector<std::vector<oid_t>> local_vertices_id;
    this->ShuffleVertex(vertices_, local_vertices_id, *partitioner);

    std::vector<oid_t> sorted_vertices;
    for (auto& buf : local_vertices_id) {
      sorted_vertices.insert(sorted_vertices.end(), buf.begin(), buf.end());
    }
    std::sort(sorted_vertices.begin(), sorted_vertices.end());

    VertexMapBuilder<oid_t, vid_t> builder(fid, fnum, std::move(partitioner),
                                           spec_.global_vertex_map,
                                           !spec_.mutable_vertex_map);
    for (auto& v : sorted_vertices) {
      builder.add_vertex(v);
    }
    vertex_map_ =
        std::unique_ptr<VertexMap<oid_t, vid_t>>(new VertexMap<oid_t, vid_t>());
    builder.finish(comm_spec_, *vertex_map_);
  }

  void AddEdge(const oid_t& src, const oid_t& dst,
               const edata_t& data) override {
    edges_src_.emplace_back(src);
    edges_dst_.emplace_back(dst);
    edges_data_.emplace_back(data);
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) override {
    if (spec_.rebalance) {
      Rebalancer<oid_t, vid_t> rebalancer(spec_.rebalance_vertex_factor,
                                          std::move(vertex_map_));
      for (auto& v : edges_src_) {
        rebalancer.inc_degree(v);
      }
      if (!spec_.directed) {
        for (auto& v : edges_dst_) {
          rebalancer.inc_degree(v);
        }
      }

      vertex_map_ = rebalancer.finish(comm_spec_);
    }

    fid_t fnum = comm_spec_.fnum();
    std::vector<ShuffleOut<vid_t, vid_t, edata_t>> edges_to_frag(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      int worker_id = comm_spec_.FragToWorker(i);
      edges_to_frag[i].Init(comm_spec_.comm(), edge_tag, 4096000);
      edges_to_frag[i].SetDestination(worker_id, i);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag[i].DisableComm();
      }
    }
    std::vector<ShuffleBufferTuple<vid_t, vid_t, edata_t>> got_edges;
    std::thread edge_recv_thread([&, this]() {
      ShuffleIn<vid_t, vid_t, edata_t> data_in;
      data_in.Init(comm_spec_.fnum(), comm_spec_.comm(), edge_tag);
      fid_t dst_fid;
      int src_worker_id;
      while (!data_in.Finished()) {
        src_worker_id = data_in.Recv(dst_fid);
        if (src_worker_id == -1) {
          break;
        }
        if (dst_fid == comm_spec_.fid()) {
          got_edges.emplace_back(std::move(data_in.buffers()));
          data_in.Clear();
        }
      }
    });

    size_t added_edges = edges_src_.size();
    for (size_t i = 0; i < added_edges; ++i) {
      vid_t src_gid, dst_gid;
      if (vertex_map_->GetGid(edges_src_[i], src_gid) &&
          vertex_map_->GetGid(edges_dst_[i], dst_gid)) {
        fid_t src_fid = id_parser_.get_fragment_id(src_gid);
        fid_t dst_fid = id_parser_.get_fragment_id(dst_gid);
        edges_to_frag[src_fid].Emplace(src_gid, dst_gid, edges_data_[i]);
        if (src_fid != dst_fid) {
          edges_to_frag[dst_fid].Emplace(src_gid, dst_gid, edges_data_[i]);
        }
      }
    }

    for (auto& ea : edges_to_frag) {
      ea.Flush();
    }
    edge_recv_thread.join();

    MPI_Barrier(comm_spec_.comm());
    got_edges.emplace_back(
        std::move(edges_to_frag[comm_spec_.fid()].buffers()));
    edges_to_frag[comm_spec_.fid()].Clear();

    std::vector<Edge<vid_t, edata_t>> processed_edges;
    for (auto& buffers : got_edges) {
      foreach_rval(buffers, [this, &processed_edges](vid_t&& src, vid_t&& dst,
                                                     edata_t&& data) {
        processed_edges.emplace_back(src, dst, std::move(data));
      });
    }

    std::vector<std::vector<oid_t>> local_vertices_id;
    std::vector<std::vector<vdata_t>> local_vertices_data;
    this->ShuffleVertexData(vertices_, vdata_, local_vertices_id,
                            local_vertices_data, vertex_map_->GetPartitioner());
    size_t buf_num = local_vertices_id.size();
    std::vector<internal::Vertex<vid_t, vdata_t>> processed_vertices;
    for (size_t buf_i = 0; buf_i < buf_num; ++buf_i) {
      std::vector<oid_t>& local_vertices = local_vertices_id[buf_i];
      std::vector<vdata_t>& local_vdata = local_vertices_data[buf_i];
      size_t local_vertices_num = local_vertices.size();
      for (size_t i = 0; i < local_vertices_num; ++i) {
        vid_t gid;
        if (vertex_map_->GetGid(local_vertices[i], gid)) {
          processed_vertices.emplace_back(gid, std::move(local_vdata[i]));
        }
      }
    }

    fragment = std::make_shared<fragment_t>();
    fragment->Init(comm_spec_.fid(), spec_.directed, std::move(vertex_map_),
                   processed_vertices, processed_edges);

    this->InitOuterVertexData(fragment);
  }

 private:
  std::vector<oid_t> vertices_;
  std::vector<vdata_t> vdata_;

  std::vector<oid_t> edges_src_;
  std::vector<oid_t> edges_dst_;
  std::vector<edata_t> edges_data_;

  std::unique_ptr<VertexMap<oid_t, vid_t>> vertex_map_;

  using BasicFragmentLoaderBase<FRAG_T>::comm_spec_;
  using BasicFragmentLoaderBase<FRAG_T>::spec_;
  using BasicFragmentLoaderBase<FRAG_T>::id_parser_;

  using BasicFragmentLoaderBase<FRAG_T>::edge_tag;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_RB_FRAGMENT_LOADER_BETA_H_