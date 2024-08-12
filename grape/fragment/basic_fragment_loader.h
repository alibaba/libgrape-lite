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
#include "grape/vertex_map/partitioner.h"
#include "grape/vertex_map/vertex_map_beta.h"
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

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_
