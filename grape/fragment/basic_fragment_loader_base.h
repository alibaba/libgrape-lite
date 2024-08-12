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

#include "grape/util.h"
#include "grape/vertex_map/idxers/idxers.h"
#include "grape/vertex_map/partitioner.h"
#include "grape/vertex_map/vertex_map.h"

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

  PartitionerType partitioner_type;
  IdxerType idxer_type;

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

  std::string to_string() const {
    std::string ret;
    ret += (directed ? "directed-" : "undirected-");
    if (rebalance) {
      ret += "rebalance-" + std::to_string(rebalance_vertex_factor) + "-";
    } else {
      ret += "no-rebalance-";
    }
    if (partitioner_type == PartitionerType::kHashPartitioner) {
      ret += "hash-partitioner-";
    } else if (partitioner_type == PartitionerType::kMapPartitioner) {
      ret += "map-partitioner-";
    } else if (partitioner_type == PartitionerType::kSegmentedPartitioner) {
      ret += "segmented-partitioner-";
    } else {
      LOG(FATAL) << "Unknown partitioner type";
    }
    if (idxer_type == IdxerType::kHashMapIdxer) {
      ret += "hashmap-idxer";
    } else if (idxer_type == IdxerType::kSortedArrayIdxer) {
      ret += "sorted-array-idxer";
    } else if (idxer_type == IdxerType::kLocalIdxer) {
      ret += "local-idxer";
    } else if (idxer_type == IdxerType::kPTHashIdxer) {
      ret += "pthash-idxer";
    } else {
      LOG(FATAL) << "Unknown idxer type";
    }
    return ret;
  }
};

inline LoadGraphSpec DefaultLoadGraphSpec() {
  LoadGraphSpec spec;
  spec.directed = true;
  spec.rebalance = true;
  spec.rebalance_vertex_factor = 0;
  spec.serialize = false;
  spec.deserialize = false;
  spec.partitioner_type = PartitionerType::kHashPartitioner;
  spec.idxer_type = IdxerType::kHashMapIdxer;
  return spec;
}

template <typename FRAG_T>
std::pair<std::string, std::string> generate_signature(
    const std::string& efile, const std::string& vfile,
    const LoadGraphSpec& spec) {
  std::string spec_info = spec.to_string();
  std::string frag_type_name = FRAG_T::type_info();
  std::string md5 = compute_hash({efile, vfile, spec_info, frag_type_name});
  std::string desc = "efile: " + efile + "\n";
  desc += "vfile: " + vfile + "\n";
  desc += "spec: " + spec_info + "\n";
  desc += "frag_type: " + frag_type_name + "\n";
  return std::make_pair(md5, desc);
}

template <typename FRAG_T, typename IOADAPTOR_T>
bool SerializeFragment(std::shared_ptr<FRAG_T>& fragment,
                       const CommSpec& comm_spec, const std::string& efile,
                       const std::string& vfile, const LoadGraphSpec& spec) {
  auto pair = generate_signature<FRAG_T>(efile, vfile, spec);
  std::string typed_prefix = spec.serialization_prefix + "/" + pair.first +
                             "/" + "part_" + std::to_string(comm_spec.fnum());
  if (!create_directories(typed_prefix)) {
    LOG(ERROR) << "Failed to create directory: " << typed_prefix << ", "
               << std::strerror(errno);
    return false;
  }
  std::string sigfile_name = typed_prefix + "/sig";
  if (exists_file(sigfile_name)) {
    LOG(ERROR) << "Signature file exists: " << sigfile_name;
    return false;
  }

  char serial_file[1024];
  snprintf(serial_file, sizeof(serial_file), "%s/%s", typed_prefix.c_str(),
           kSerializationVertexMapFilename);
  fragment->GetVertexMap().template Serialize<IOADAPTOR_T>(typed_prefix,
                                                           comm_spec);
  fragment->template Serialize<IOADAPTOR_T>(typed_prefix);

  MPI_Barrier(comm_spec.comm());
  if (comm_spec.worker_id() == 0) {
    std::ofstream sigfile(sigfile_name);
    if (!sigfile.is_open()) {
      LOG(ERROR) << "Failed to open signature file: " << sigfile_name;
      return false;
    }
    sigfile << pair.second;
  }

  return true;
}

template <typename FRAG_T, typename IOADAPTOR_T>
bool DeserializeFragment(std::shared_ptr<FRAG_T>& fragment,
                         const CommSpec& comm_spec, const std::string& efile,
                         const std::string& vfile, const LoadGraphSpec& spec) {
  auto pair = generate_signature<FRAG_T>(efile, vfile, spec);
  std::string typed_prefix = spec.deserialization_prefix + "/" + pair.first +
                             "/" + "part_" + std::to_string(comm_spec.fnum());
  std::string sigfile_name = typed_prefix + "/sig";
  if (!exists_file(sigfile_name)) {
    LOG(ERROR) << "Signature file not exists: " << sigfile_name;
    return false;
  }
  std::string sigfile_content;
  std::ifstream sigfile(sigfile_name);
  if (!sigfile.is_open()) {
    LOG(ERROR) << "Failed to open signature file: " << sigfile_name;
    return false;
  }
  std::string line;
  while (std::getline(sigfile, line)) {
    sigfile_content += line + "\n";
  }
  if (sigfile_content != pair.second) {
    LOG(ERROR) << "Signature mismatch: " << sigfile_content
               << ", expected: " << pair.second;
    return false;
  }

  auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(typed_prefix));
  if (io_adaptor->IsExist()) {
    std::unique_ptr<VertexMap<typename FRAG_T::oid_t, typename FRAG_T::vid_t>>
        vm_ptr(new VertexMap<typename FRAG_T::oid_t, typename FRAG_T::vid_t>());
    vm_ptr->template Deserialize<IOADAPTOR_T>(typed_prefix, comm_spec);
    fragment = std::shared_ptr<FRAG_T>(new FRAG_T());
    fragment->template Deserialize<IOADAPTOR_T>(std::move(vm_ptr), typed_prefix,
                                                comm_spec.fid());
    return true;
  } else {
    return false;
  }
}

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
  virtual ~BasicFragmentLoaderBase() {}

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
    size_t added_vertices = added_vertices_id.size();
    for (size_t i = 0; i < added_vertices; ++i) {
      fid_t dst_fid = partitioner.GetPartitionId(added_vertices_id[i]);
      if (dst_fid == fnum) {
        LOG(ERROR) << "Unknown partition id for vertex "
                   << added_vertices_id[i];
      } else {
        partitioned_vertices_out[dst_fid].emplace_back(
            std::move(added_vertices_id[i]));
      }
    }

    local_vertices_id.emplace_back(std::move(partitioned_vertices_out[fid]));

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
          sync_comm::Recv(recv_vertices, src_worker_id, vertex_tag,
                          comm_spec_.comm());
          local_vertices_id.emplace_back(std::move(recv_vertices));
        }
        src_worker_id = (src_worker_id + comm_spec_.worker_num() - 1) %
                        comm_spec_.worker_num();
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
      if (dst_fid == fnum) {
        LOG(ERROR) << "Unknown partition id for vertex "
                   << added_vertices_id[i];
      } else {
        partitioned_vertices_out[dst_fid].emplace_back(
            std::move(added_vertices_id[i]));
        partitioned_vdata_out[dst_fid].emplace_back(
            std::move(added_vertices_data[i]));
      }
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
