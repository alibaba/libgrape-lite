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

#ifndef GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_BETA_H_
#define GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_BETA_H_

#include "grape/fragment/basic_fragment_loader.h"
#include "grape/fragment/basic_fragment_loader_base.h"
#include "grape/fragment/rebalancer.h"
#include "grape/graph/vertex.h"
#include "grape/vertex_map/vertex_map_beta.h"

namespace grape {

template <typename FRAG_T>
class BasicFragmentLoaderBeta : public BasicFragmentLoaderBase<FRAG_T> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using internal_oid_t = typename InternalOID<oid_t>::type;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

 public:
  explicit BasicFragmentLoaderBeta(const CommSpec& comm_spec,
                                   const LoadGraphSpec& spec)
      : BasicFragmentLoaderBase<FRAG_T>(comm_spec, spec) {
    if (!spec_.global_vertex_map) {
      LOG(ERROR) << "Global vertex map is required in BasicFragmentLoaderBeta";
      spec_.global_vertex_map = true;
    }
    if (spec_.rebalance) {
      LOG(ERROR) << "Rebalance is not supported in BasicFragmentLoaderBeta";
      spec_.rebalance = false;
    }
    recv_thread_running_ = false;
  }

  ~BasicFragmentLoaderBeta() {
    if (recv_thread_running_) {
      for (auto& ea : edges_to_frag_) {
        ea.Flush();
      }
      edge_recv_thread_.join();
    }
  }

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
    std::vector<std::vector<vdata_t>> local_vertices_data;
    this->ShuffleVertexData(vertices_, vdata_, local_vertices_id,
                            local_vertices_data, *partitioner);
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

    for (size_t buf_i = 0; buf_i < local_vertices_id.size(); ++buf_i) {
      std::vector<oid_t>& local_vertices = local_vertices_id[buf_i];
      std::vector<vdata_t>& local_vdata = local_vertices_data[buf_i];
      size_t local_vertices_num = local_vertices.size();
      for (size_t i = 0; i < local_vertices_num; ++i) {
        vid_t gid;
        if (vertex_map_->GetGid(local_vertices[i], gid)) {
          processed_vertices_.emplace_back(gid, std::move(local_vdata[i]));
        }
      }
    }

    edges_to_frag_.resize(fnum);
    for (fid_t fid = 0; fid < fnum; ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag, 4096000);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag_[fid].DisableComm();
      }
    }
    edge_recv_thread_ =
        std::thread(&BasicFragmentLoaderBeta::edgeRecvRoutine, this);
    recv_thread_running_ = true;
  }

  void AddEdge(const oid_t& src, const oid_t& dst,
               const edata_t& data) override {
    vid_t src_gid, dst_gid;
    if (vertex_map_->GetGid(src, src_gid) &&
        vertex_map_->GetGid(dst, dst_gid)) {
      fid_t src_fid = id_parser_.get_fragment_id(src_gid);
      fid_t dst_fid = id_parser_.get_fragment_id(dst_gid);
      edges_to_frag_[src_fid].Emplace(src_gid, dst_gid, data);
      if (src_fid != dst_fid) {
        edges_to_frag_[dst_fid].Emplace(src_gid, dst_gid, data);
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

    std::vector<Edge<vid_t, edata_t>> processed_edges;
    for (auto& buffers : got_edges_) {
      foreach_rval(buffers, [this, &processed_edges](vid_t&& src, vid_t&& dst,
                                                     edata_t&& data) {
        processed_edges.emplace_back(src, dst, std::move(data));
      });
    }

    fragment = std::make_shared<fragment_t>();
    fragment->Init(comm_spec_.fid(), spec_.directed, std::move(vertex_map_),
                   processed_vertices_, processed_edges);

    this->InitOuterVertexData(fragment);
  }

 private:
  void edgeRecvRoutine() {
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
        got_edges_.emplace_back(std::move(data_in.buffers()));
        data_in.Clear();
      }
    }
  }

  std::vector<oid_t> vertices_;
  std::vector<vdata_t> vdata_;

  std::vector<internal::Vertex<vid_t, vdata_t>> processed_vertices_;

  std::unique_ptr<VertexMap<oid_t, vid_t>> vertex_map_;

  std::vector<ShuffleOut<vid_t, vid_t, edata_t>> edges_to_frag_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  std::vector<ShuffleBufferTuple<vid_t, vid_t, edata_t>> got_edges_;

  std::vector<vid_t> src_gid_list_;
  std::vector<vid_t> dst_gid_list_;
  std::vector<edata_t> edata_;

  using BasicFragmentLoaderBase<FRAG_T>::comm_spec_;
  using BasicFragmentLoaderBase<FRAG_T>::spec_;
  using BasicFragmentLoaderBase<FRAG_T>::id_parser_;

  using BasicFragmentLoaderBase<FRAG_T>::edge_tag;
};

};  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_BETA_H_
