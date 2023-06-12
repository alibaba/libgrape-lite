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

#ifndef GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_

#include <stddef.h>

#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "grape/communication/shuffle.h"
#include "grape/config.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/util.h"
#include "grape/utils/concurrent_queue.h"
#include "grape/utils/vertex_array.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief LoadGraphSpec determines the specification to load a graph.
 *
 */
struct LoadGraphSpec {
  bool directed;
  bool rebalance;
  int rebalance_vertex_factor;

  bool serialize;
  std::string serialization_prefix;

  bool deserialize;
  std::string deserialization_prefix;

  void set_directed(bool val = true) { directed = val; }
  void set_rebalance(bool flag, int weight) {
    rebalance = flag;
    rebalance_vertex_factor = weight;
  }

  void set_serialize(bool flag, const std::string& prefix) {
    serialize = flag;
    serialization_prefix = prefix;
  }

  void set_deserialize(bool flag, const std::string& prefix) {
    deserialize = flag;
    deserialization_prefix = prefix;
  }
};

inline LoadGraphSpec DefaultLoadGraphSpec() {
  LoadGraphSpec spec;
  spec.directed = true;
  spec.rebalance = true;
  spec.rebalance_vertex_factor = 0;
  spec.serialize = false;
  spec.deserialize = false;
  return spec;
}

template <typename FRAG_T, typename IOADAPTOR_T>
class BasicFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using internal_oid_t = typename InternalOID<oid_t>::type;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using vertex_map_t = typename fragment_t::vertex_map_t;
  using partitioner_t = typename vertex_map_t::partitioner_t;

  static constexpr LoadStrategy load_strategy = fragment_t::load_strategy;

 public:
  explicit BasicFragmentLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec) {
    comm_spec_.Dup();
    vm_ptr_ = std::make_shared<vertex_map_t>(comm_spec_);
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

    recv_thread_running_ = false;
  }

  ~BasicFragmentLoader() { Stop(); }

  void SetPartitioner(const partitioner_t& partitioner) {
    vm_ptr_->SetPartitioner(partitioner);
  }

  void SetPartitioner(partitioner_t&& partitioner) {
    vm_ptr_->SetPartitioner(std::move(partitioner));
  }

  void Start() {
    vertex_recv_thread_ =
        std::thread(&BasicFragmentLoader::vertexRecvRoutine, this);
    edge_recv_thread_ =
        std::thread(&BasicFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;
  }

  void Stop() {
    if (recv_thread_running_) {
      for (auto& va : vertices_to_frag_) {
        va.Flush();
      }
      for (auto& ea : edges_to_frag_) {
        ea.Flush();
      }
      vertex_recv_thread_.join();
      edge_recv_thread_.join();
      recv_thread_running_ = false;
    }
  }

  void AddVertex(const oid_t& id, const vdata_t& data) {
    internal_oid_t internal_id(id);
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t fid = partitioner.GetPartitionId(internal_id);
    vertices_to_frag_[fid].Emplace(internal_id, data);
  }

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    internal_oid_t internal_src(src);
    internal_oid_t internal_dst(dst);
    auto& partitioner = vm_ptr_->GetPartitioner();
    fid_t src_fid = partitioner.GetPartitionId(internal_src);
    fid_t dst_fid = partitioner.GetPartitionId(internal_dst);
    edges_to_frag_[src_fid].Emplace(internal_src, internal_dst, data);
    if (src_fid != dst_fid) {
      edges_to_frag_[dst_fid].Emplace(internal_src, internal_dst, data);
    }
  }

  bool SerializeFragment(std::shared_ptr<fragment_t>& fragment,
                         const std::string& serialization_prefix) {
    std::string type_prefix = fragment_t::type_info();
    std::string typed_prefix = serialization_prefix + "/" + type_prefix;
    char serial_file[1024];
    snprintf(serial_file, sizeof(serial_file), "%s/%s", typed_prefix.c_str(),
             kSerializationVertexMapFilename);
    vm_ptr_->template Serialize<IOADAPTOR_T>(typed_prefix);
    fragment->template Serialize<IOADAPTOR_T>(typed_prefix);

    return true;
  }

  bool existSerializationFile(const std::string& prefix) {
    char vm_fbuf[1024], frag_fbuf[1024];
    snprintf(vm_fbuf, sizeof(vm_fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);
    snprintf(frag_fbuf, sizeof(frag_fbuf), kSerializationFilenameFormat,
             prefix.c_str(), comm_spec_.fid());
    std::string vm_path = vm_fbuf;
    std::string frag_path = frag_fbuf;
    return exists_file(vm_path) && exists_file(frag_path);
  }

  bool DeserializeFragment(std::shared_ptr<fragment_t>& fragment,
                           const std::string& deserialization_prefix) {
    std::string type_prefix = fragment_t::type_info();
    std::string typed_prefix = deserialization_prefix + "/" + type_prefix;
    if (!existSerializationFile(typed_prefix)) {
      return false;
    }
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(typed_prefix));
    if (io_adaptor->IsExist()) {
      vm_ptr_->template Deserialize<IOADAPTOR_T>(typed_prefix,
                                                 comm_spec_.fid());
      fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr_));
      fragment->template Deserialize<IOADAPTOR_T>(typed_prefix,
                                                  comm_spec_.fid());
      return true;
    } else {
      return false;
    }
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment, bool directed) {
    for (auto& va : vertices_to_frag_) {
      va.Flush();
    }
    for (auto& ea : edges_to_frag_) {
      ea.Flush();
    }
    vertex_recv_thread_.join();
    edge_recv_thread_.join();
    recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());

    got_vertices_.emplace_back(
        std::move(vertices_to_frag_[comm_spec_.fid()].buffers()));
    vertices_to_frag_[comm_spec_.fid()].Clear();
    got_edges_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].buffers()));
    edges_to_frag_[comm_spec_.fid()].Clear();

    vm_ptr_->Init();
    auto builder = vm_ptr_->GetLocalBuilder();
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
    builder.finish(*vm_ptr_);

    processed_vertices_.clear();
    if (!std::is_same<vdata_t, EmptyType>::value) {
      for (auto& buffers : got_vertices_) {
        foreach_rval(buffers, [this](internal_oid_t&& id, vdata_t&& data) {
          vid_t gid;
          CHECK(vm_ptr_->_GetGid(id, gid));
          processed_vertices_.emplace_back(gid, std::move(data));
        });
      }
    }
    got_vertices_.clear();

    for (auto& buffers : got_edges_) {
      foreach_rval(buffers, [this](internal_oid_t&& src, internal_oid_t&& dst,
                                   edata_t&& data) {
        vid_t src_gid, dst_gid;
        CHECK(vm_ptr_->_GetGid(src, src_gid));
        CHECK(vm_ptr_->_GetGid(dst, dst_gid));
        processed_edges_.emplace_back(src_gid, dst_gid, std::move(data));
      });
    }

    fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr_));
    fragment->Init(comm_spec_.fid(), directed, processed_vertices_,
                   processed_edges_);

    if (!std::is_same<vdata_t, EmptyType>::value) {
      initOuterVertexData(fragment);
    }
  }

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

  void initOuterVertexData(std::shared_ptr<fragment_t> fragment) {
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

 private:
  CommSpec comm_spec_;
  std::shared_ptr<vertex_map_t> vm_ptr_;

  std::vector<ShuffleOut<internal_oid_t, vdata_t>> vertices_to_frag_;
  std::vector<ShuffleOut<internal_oid_t, internal_oid_t, edata_t>>
      edges_to_frag_;

  std::thread vertex_recv_thread_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  std::vector<ShuffleBufferTuple<internal_oid_t, vdata_t>> got_vertices_;
  std::vector<ShuffleBufferTuple<internal_oid_t, internal_oid_t, edata_t>>
      got_edges_;

  std::vector<internal::Vertex<vid_t, vdata_t>> processed_vertices_;
  std::vector<Edge<vid_t, edata_t>> processed_edges_;

  static constexpr int vertex_tag = 5;
  static constexpr int edge_tag = 6;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_
