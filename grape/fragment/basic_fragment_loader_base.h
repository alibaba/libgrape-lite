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

#ifndef GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_BASE_H_
#define GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_BASE_H_

namespace grape {

template <typename FRAG_T>
class BasicFragmentLoaderBase {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;
  using internal_oid_t = typename InternalOID<oid_t>::type;

 public:
  BasicFragmentLoaderBase(const CommSpec& comm_spec, const LoadGraphSpec& spec)
      : comm_spec_(comm_spec), spec_(spec) {
    comm_spec_.Dup();
    id_parser_.init(comm_spec_.fnum());
  }

  virtual void AddVertex(const oid_t& id, const vdata_t& data) = 0;
  virtual void ConstructVertices() = 0;
  virtual void AddEdge(const oid_t& src, const oid_t& dst,
                       const edata_t& data) = 0;
  virtual void ConstructFragment(std::shared_ptr<fragment_t>& fragment) = 0;

 protected:
  void InitOuterVertexData(std::shared_ptr<fragment_t> fragment) {
    int worker_num = comm_spec_.worker_num();

    std::vector<std::vector<vid_t>> request_gid_lists(worker_num);
    auto& outer_vertices = fragment->OuterVertices();
    for (auto& v : outer_vertices) {
      fid_t fid = fragment->GetFragId(v);
      request_gid_lists[comm_spec_.FragToWorker(fid)].emplace_back(
          fragment->GetOuterVertexGid(v));
    }
    std::vector<std::vector<vid_t>> requested_gid_lists(worker_num);
    sync_comm::AllToAll(request_gid_lists, requested_gid_lists,
                        comm_spec_.comm());
    std::vector<std::vector<vdata_t>> response_vdata_lists(worker_num);
    for (int i = 0; i < worker_num; ++i) {
      auto& id_vec = requested_gid_lists[i];
      auto& data_vec = response_vdata_lists[i];
      data_vec.reserve(id_vec.size());
      for (auto id : id_vec) {
        typename fragment_t::vertex_t v;
        CHECK(fragment->InnerVertexGid2Vertex(id, v));
        data_vec.emplace_back(fragment->GetData(v));
      }
    }
    std::vector<std::vector<vdata_t>> responsed_vdata_lists(worker_num);
    sync_comm::AllToAll(response_vdata_lists, responsed_vdata_lists,
                        comm_spec_.comm());
    for (int i = 0; i < worker_num; ++i) {
      auto& id_vec = request_gid_lists[i];
      auto& data_vec = responsed_vdata_lists[i];
      CHECK_EQ(id_vec.size(), data_vec.size());
      size_t num = id_vec.size();
      for (size_t k = 0; k < num; ++k) {
        typename fragment_t::vertex_t v;
        CHECK(fragment->OuterVertexGid2Vertex(id_vec[k], v));
        fragment->SetData(v, data_vec[k]);
      }
    }
  }

  void ShuffleVertex(const std::vector<oid_t>& added_vertices_id,
                     std::vector<std::vector<oid_t>>& local_vertices_id,
                     const IPartitioner<oid_t>& partitioner) {
    fid_t fnum = comm_spec_.fnum();
    fid_t fid = comm_spec_.fid();
    std::vector<std::vector<oid_t>> partitioned_vertices_out(fnum);
    std::vector<std::vector<vdata_t>> partitioned_vdata_out(fnum);
    size_t added_vertices = added_vertices_id.size();
    for (size_t i = 0; i < added_vertices; ++i) {
      fid_t dst_fid = partitioner.GetPartitionId(added_vertices_id[i]);
      partitioned_vertices_out[dst_fid].emplace_back(
          std::move(added_vertices_id[i]));
    }

    std::vector<int64_t> partitioned_vertices_size_out(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      partitioned_vertices_size_out[i] = partitioned_vertices_out[i].size();
    }

    local_vertices_id.emplace_back(std::move(partitioned_vertices_out[fid]));

    std::thread send_thread([&]() {
      int dst_worker_id =
          (comm_spec_.worker_id() + 1) % comm_spec_.worker_num();
      while (dst_worker_id != comm_spec_.worker_id()) {
        for (fid_t fid = 0; fid < fnum; ++fid) {
          if (comm_spec_.FragToWorker(fid) != comm_spec_.worker_id()) {
            continue;
          }
          sync_comm::Send(partitioned_vertices_out[fid], dst_worker_id,
                          vertex_tag, comm_spec_.comm());
        }
        dst_worker_id = (dst_worker_id + 1) % comm_spec_.worker_num();
      }
    });
    std::thread recv_thread([&]() {
      int src_worker_id =
          (comm_spec_.worker_id() + comm_spec_.worker_num() - 1) %
          comm_spec_.worker_num();
      while (src_worker_id != comm_spec_.worker_id()) {
        for (fid_t fid = 0; fid < fnum; ++fid) {
          if (comm_spec_.FragToWorker(fid) != src_worker_id) {
            continue;
          }
          std::vector<oid_t> recv_vertices;
          sync_comm::Recv(recv_vertices, src_worker_id, vertex_tag,
                          comm_spec_.comm());
          local_vertices_id.emplace_back(std::move(recv_vertices));
        }
      }
    });

    recv_thread.join();
    send_thread.join();
  }

  void ShuffleVertexData(const std::vector<oid_t>& added_vertices_id,
                         const std::vector<vdata_t>& added_vertices_data,
                         std::vector<std::vector<oid_t>>& local_vertices_id,
                         std::vector<std::vector<vdata_t>>& local_vertices_data,
                         const IPartitioner<oid_t>& partitioner) {
    fid_t fnum = comm_spec_.fnum();
    fid_t fid = comm_spec_.fid();
    std::vector<std::vector<oid_t>> partitioned_vertices_out(fnum);
    std::vector<std::vector<vdata_t>> partitioned_vdata_out(fnum);
    size_t added_vertices = added_vertices_id.size();
    for (size_t i = 0; i < added_vertices; ++i) {
      fid_t dst_fid = partitioner.GetPartitionId(added_vertices_id[i]);
      partitioned_vertices_out[dst_fid].emplace_back(
          std::move(added_vertices_id[i]));
      partitioned_vdata_out[dst_fid].emplace_back(
          std::move(added_vertices_data[i]));
    }

    std::vector<int64_t> partitioned_vertices_size_out(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      partitioned_vertices_size_out[i] = partitioned_vertices_out[i].size();
    }

    local_vertices_id.emplace_back(std::move(partitioned_vertices_out[fid]));
    local_vertices_data.emplace_back(std::move(partitioned_vdata_out[fid]));

    std::thread send_thread([&]() {
      int dst_worker_id =
          (comm_spec_.worker_id() + 1) % comm_spec_.worker_num();
      while (dst_worker_id != comm_spec_.worker_id()) {
        for (fid_t fid = 0; fid < fnum; ++fid) {
          if (comm_spec_.FragToWorker(fid) != dst_worker_id) {
            continue;
          }
          sync_comm::Send(partitioned_vertices_out[fid], dst_worker_id,
                          vertex_tag, comm_spec_.comm());
          sync_comm::Send(partitioned_vdata_out[fid], dst_worker_id, vertex_tag,
                          comm_spec_.comm());
        }
        dst_worker_id = (dst_worker_id + 1) % comm_spec_.worker_num();
      }
    });
    std::thread recv_thread([&]() {
      int src_worker_id =
          (comm_spec_.worker_id() + comm_spec_.worker_num() - 1) %
          comm_spec_.worker_num();
      while (src_worker_id != comm_spec_.worker_id()) {
        for (fid_t fid = 0; fid < fnum; ++fid) {
          if (comm_spec_.FragToWorker(fid) != comm_spec_.worker_id()) {
            continue;
          }
          std::vector<oid_t> recv_vertices;
          std::vector<vdata_t> recv_vdata;
          sync_comm::Recv(recv_vertices, src_worker_id, vertex_tag,
                          comm_spec_.comm());
          sync_comm::Recv(recv_vdata, src_worker_id, vertex_tag,
                          comm_spec_.comm());
          local_vertices_id.emplace_back(std::move(recv_vertices));
          local_vertices_data.emplace_back(std::move(recv_vdata));
        }

        src_worker_id = (src_worker_id + comm_spec_.worker_num() - 1) %
                        comm_spec_.worker_num();
      }
    });

    recv_thread.join();
    send_thread.join();
  }

  CommSpec comm_spec_;
  LoadGraphSpec spec_;
  IdParser<vid_t> id_parser_;

  static constexpr int vertex_tag = 5;
  static constexpr int edge_tag = 6;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_BASE_H_
