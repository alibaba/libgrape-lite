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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/fragment/mutable_edgecut_fragment.h>
#include <grape/grape.h>
#include <grape/util.h>

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(mutable_efile_base, "", "base of mutable edge file");
DEFINE_string(mutable_efile_delta, "", "delta of mutable edge file");
DEFINE_string(serialization_prefix, "",
              "directory to place serialization files");

void Init() {
  if (FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input edge files.";
  }
  if (access(FLAGS_serialization_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_serialization_prefix.c_str(), 0777);
  }

  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == grape::kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  grape::FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename OID_T, typename VID_T>
bool verify_vertex_map(const grape::CommSpec& comm_spec,
                       const grape::VertexMap<OID_T, VID_T>& vertex_map) {
  grape::fid_t fnum = comm_spec.fnum();
  std::vector<std::vector<std::pair<VID_T, OID_T>>> all_maps_g2o(fnum);
  std::vector<std::vector<std::pair<OID_T, VID_T>>> all_maps_o2g(fnum);
  bool ret = true;
  for (grape::fid_t fid = 0; fid != fnum; ++fid) {
    VID_T frag_vnum = vertex_map.GetInnerVertexSize(fid);
    for (VID_T lid = 0; lid < frag_vnum; ++lid) {
      OID_T oid_a, oid_b;
      if (vertex_map.GetOid(fid, lid, oid_a)) {
        VID_T gid_a{}, gid_b{};
        if (!vertex_map.GetGid(fid, oid_a, gid_a)) {
          LOG(ERROR) << "Vertex " << oid_a << " not found by fid+oid in vertex "
                     << "map.";
          ret = false;
          continue;
        }
        if (!vertex_map.GetGid(oid_a, gid_b)) {
          LOG(ERROR) << "Vertex " << oid_a << " not found by oid in vertex "
                     << "map.";
          ret = false;
          continue;
        }
        if (gid_a != gid_b) {
          LOG(ERROR) << "Vertex " << oid_a << " gid not consistent.";
          ret = false;
          continue;
        }
        if (!vertex_map.GetOid(gid_a, oid_b)) {
          LOG(ERROR) << "Vertex " << gid_a << " not found by gid in vertex "
                     << "map.";
          ret = false;
          continue;
        }
        if (oid_a != oid_b) {
          LOG(ERROR) << "Vertex " << gid_a << " oid not consistent.";
          ret = false;
          continue;
        }
        all_maps_g2o[gid_a % fnum].emplace_back(gid_a, oid_a);
        all_maps_o2g[std::hash<OID_T>{}(oid_a) % fnum].emplace_back(oid_a,
                                                                    gid_a);
      }
    }
  }

  {
    std::vector<std::vector<std::pair<VID_T, OID_T>>> all_maps_g2o_in(fnum);
    grape::sync_comm::AllToAll(all_maps_g2o, all_maps_g2o_in, comm_spec.comm());

    std::vector<std::pair<VID_T, OID_T>> all_maps_merged;
    for (auto& maps : all_maps_g2o_in) {
      all_maps_merged.insert(all_maps_merged.end(), maps.begin(), maps.end());
    }

    std::sort(all_maps_merged.begin(), all_maps_merged.end());
    for (size_t i = 1; i < all_maps_merged.size(); ++i) {
      if (all_maps_merged[i].first == all_maps_merged[i - 1].first) {
        if (all_maps_merged[i].second != all_maps_merged[i - 1].second) {
          LOG(ERROR) << "Vertex " << all_maps_merged[i].first
                     << " has different oid in different fragments.";
          ret = false;
        }
      }
    }
  }

  {
    std::vector<std::vector<std::pair<OID_T, VID_T>>> all_maps_o2g_in(fnum);
    grape::sync_comm::AllToAll(all_maps_o2g, all_maps_o2g_in, comm_spec.comm());

    std::vector<std::pair<OID_T, VID_T>> all_maps_merged;
    for (auto& maps : all_maps_o2g_in) {
      all_maps_merged.insert(all_maps_merged.end(), maps.begin(), maps.end());
    }

    std::sort(all_maps_merged.begin(), all_maps_merged.end());
    for (size_t i = 1; i < all_maps_merged.size(); ++i) {
      if (all_maps_merged[i].first == all_maps_merged[i - 1].first) {
        if (all_maps_merged[i].second != all_maps_merged[i - 1].second) {
          LOG(ERROR) << "Vertex " << all_maps_merged[i].first
                     << " has different gid in different fragments.";
          ret = false;
        }
      }
    }
  }

  return ret;
}

template <typename FRAG_T, typename VERTEX_MAP_T>
bool verify_fragment_vertex_map(const FRAG_T& frag,
                                const VERTEX_MAP_T& vertex_map) {
  auto inner_vertices = frag.InnerVertices();
  auto outer_vertices = frag.OuterVertices();
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  for (auto v : inner_vertices) {
    vid_t gid = frag.GetInnerVertexGid(v);
    oid_t oid;
    if (!vertex_map.GetOid(gid, oid)) {
      LOG(ERROR) << "Vertex " << gid << " not found in vertex map.";
      return false;
    }
  }
  for (auto v : outer_vertices) {
    vid_t gid = frag.GetOuterVertexGid(v);
    oid_t oid;
    if (!vertex_map.GetOid(gid, oid)) {
      LOG(ERROR) << "Vertex " << gid << " not found in vertex map.";
      return false;
    }
  }
  return true;
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void test_build_vertex_map(const std::string& efile, const std::string& vfile,
                           const grape::LoadGraphSpec& graph_spec,
                           const grape::CommSpec& comm_spec) {
  using FRAG_T =
      grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T>;
  std::shared_ptr<FRAG_T> fragment =
      grape::LoadGraph<FRAG_T>(efile, vfile, comm_spec, graph_spec);

  verify_fragment_vertex_map(*fragment, fragment->GetVertexMap());
  verify_vertex_map(comm_spec, fragment->GetVertexMap());
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void test_mutate_vertex_map(const std::string& efile_base,
                            const std::string& vfile,
                            const std::string& efile_delta,
                            const grape::LoadGraphSpec& graph_spec,
                            const grape::CommSpec& comm_spec) {
  using FRAG_T = grape::MutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T>;
  std::shared_ptr<FRAG_T> fragment = grape::LoadGraphAndMutate<FRAG_T>(
      efile_base, vfile, efile_delta, "", comm_spec, graph_spec);

  verify_fragment_vertex_map(*fragment, fragment->GetVertexMap());
  verify_vertex_map(comm_spec, fragment->GetVertexMap());
}

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;
  grape::gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./run_app [grape_opts]");
  if (argc == 1) {
    grape::gflags::ShowUsageWithFlagsRestrict(argv[0], "vertex_map_tests");
    exit(1);
  }
  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  grape::gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("vertex_map_tests");
  google::InstallFailureSignalHandler();

  Init();

  std::vector<bool> string_id_options({false, true});
  std::vector<bool> rebalance_options({false, true});
  std::vector<grape::PartitionerType> partitioner_options(
      {grape::PartitionerType::kHashPartitioner,
       grape::PartitionerType::kMapPartitioner,
       grape::PartitionerType::kSegmentedPartitioner});
  std::vector<grape::IdxerType> idxer_options(
      {grape::IdxerType::kHashMapIdxer, grape::IdxerType::kHashMapIdxerView,
       grape::IdxerType::kPTHashIdxer, grape::IdxerType::kSortedArrayIdxer,
       grape::IdxerType::kLocalIdxer});

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);
    int idx = 0;
    for (auto string_id : string_id_options) {
      for (auto rebalance : rebalance_options) {
        for (auto partitioner_type : partitioner_options) {
          for (auto idxer_type : idxer_options) {
            if (rebalance) {
              if (partitioner_type ==
                  grape::PartitionerType::kHashPartitioner) {
                continue;
              }
            }
            if (idxer_type == grape::IdxerType::kLocalIdxer) {
              if (partitioner_type !=
                  grape::PartitionerType::kHashPartitioner) {
                continue;
              }
            }
            bool vm_extendable =
                (idxer_type == grape::IdxerType::kHashMapIdxer ||
                 idxer_type == grape::IdxerType::kLocalIdxer);
            VLOG(2) << "Test " << idx++ << ": string_id=" << string_id
                    << ", rebalance=" << rebalance
                    << ", partitioner_type=" << partitioner_type
                    << ", idxer_type=" << idxer_type;
            grape::LoadGraphSpec graph_spec = grape::DefaultLoadGraphSpec();
            graph_spec.set_directed(false);
            if (rebalance) {
              graph_spec.set_rebalance(true, 0);
            } else {
              graph_spec.set_rebalance(false, 0);
            }
            graph_spec.partitioner_type = partitioner_type;
            graph_spec.idxer_type = idxer_type;

            graph_spec.set_serialize(true, FLAGS_serialization_prefix);
            if (string_id) {
              test_build_vertex_map<std::string, uint32_t, grape::EmptyType,
                                    double>(FLAGS_efile, FLAGS_vfile,
                                            graph_spec, comm_spec);
              if (vm_extendable) {
                test_mutate_vertex_map<std::string, uint32_t, grape::EmptyType,
                                       double>(
                    FLAGS_mutable_efile_base, FLAGS_vfile,
                    FLAGS_mutable_efile_delta, graph_spec, comm_spec);
              }
            } else {
              test_build_vertex_map<int64_t, uint32_t, grape::EmptyType,
                                    double>(FLAGS_efile, FLAGS_vfile,
                                            graph_spec, comm_spec);
              if (vm_extendable) {
                test_mutate_vertex_map<int64_t, uint32_t, grape::EmptyType,
                                       double>(
                    FLAGS_mutable_efile_base, FLAGS_vfile,
                    FLAGS_mutable_efile_delta, graph_spec, comm_spec);
              }
            }

            graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
            if (string_id) {
              test_build_vertex_map<std::string, uint32_t, grape::EmptyType,
                                    double>(FLAGS_efile, FLAGS_vfile,
                                            graph_spec, comm_spec);
              if (vm_extendable) {
                test_mutate_vertex_map<std::string, uint32_t, grape::EmptyType,
                                       double>(
                    FLAGS_mutable_efile_base, FLAGS_vfile,
                    FLAGS_mutable_efile_delta, graph_spec, comm_spec);
              }
            } else {
              test_build_vertex_map<int64_t, uint32_t, grape::EmptyType,
                                    double>(FLAGS_efile, FLAGS_vfile,
                                            graph_spec, comm_spec);
              if (vm_extendable) {
                test_mutate_vertex_map<int64_t, uint32_t, grape::EmptyType,
                                       double>(
                    FLAGS_mutable_efile_base, FLAGS_vfile,
                    FLAGS_mutable_efile_delta, graph_spec, comm_spec);
              }
            }
          }
        }
      }
    }
  }

  Finalize();

  google::ShutdownGoogleLogging();
}
