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

#include <iostream>
#include <random>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <grape/grape.h>
#include <grape/vertex_map/global_vertex_map.h>
#include <grape/vertex_map/imm_global_vertex_map.h>
#include <grape/vertex_map/ph_global_vertex_map.h>
#include "grape/io/line_parser_base.h"
#include "grape/io/local_io_adaptor.h"
#include "grape/io/tsv_line_parser.h"
#include "grape/types.h"

DEFINE_string(vfile, "", "vertex file");
DEFINE_string(out_prefix, "", "output directory of results");

void Init() {
  if (FLAGS_out_prefix.empty()) {
    LOG(FATAL) << "Please assign an output prefix.";
  }
  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
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

template <typename VM_T, typename IOADAPTOR_T = grape::LocalIOAdaptor,
          typename LINE_PARSER_T = grape::TSVLineParser<
              typename VM_T::oid_t, grape::EmptyType, grape::EmptyType>>
void LoadVertexMap(std::vector<typename VM_T::oid_t>& id_list, VM_T& vm) {
  using oid_t = typename VM_T::oid_t;
  const grape::CommSpec& comm_spec = vm.GetCommSpec();
  if (!FLAGS_vfile.empty()) {
    id_list.clear();
    auto io_adaptor = std::unique_ptr<grape::LocalIOAdaptor>(
        new grape::LocalIOAdaptor(FLAGS_vfile));
    LINE_PARSER_T line_parser;
    oid_t vertex_id;
    grape::EmptyType v_data;
    io_adaptor->Open();
    std::string line;
    while (io_adaptor->ReadLine(line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      line_parser.LineParserForVFile(line, vertex_id, v_data);
      id_list.push_back(vertex_id);
    }

    using partitioner_t = typename VM_T::partitioner_t;
    {
      partitioner_t partitioner(comm_spec.fnum(), id_list);
      vm.SetPartitioner(std::move(partitioner));
    }

    auto builder = vm.GetLocalBuilder();
    for (auto& id : id_list) {
      builder.add_vertex(id);
    }
    builder.finish(vm);
  }
}

template <typename VM_T>
void ValidateVertexMap(const std::vector<typename VM_T::oid_t>& id_list,
                       const VM_T& vm) {
  using oid_t = typename VM_T::oid_t;
  CHECK_EQ(id_list.size(), vm.GetTotalVertexSize());
  using vid_t = typename VM_T::vid_t;
  std::vector<vid_t> gid_list;
  for (auto& oid : id_list) {
    vid_t gid;
    CHECK(vm.GetGid(oid, gid));
    gid_list.push_back(gid);
    oid_t new_oid;
    CHECK(vm.GetOid(gid, new_oid));
    CHECK_EQ(oid, new_oid);
  }

  const grape::CommSpec& comm_spec = vm.GetCommSpec();
  int worker_id = comm_spec.worker_id();
  int worker_num = comm_spec.worker_num();

  if (worker_num == 1) {
    return;
  }
  if (worker_id == 0) {
    for (int dst_worker_id = 1; dst_worker_id < worker_num; ++dst_worker_id) {
      grape::sync_comm::Send(gid_list, dst_worker_id, 0, comm_spec.comm());
    }
  } else {
    std::vector<vid_t> got_gid_list;
    grape::sync_comm::Recv(got_gid_list, 0, 0, comm_spec.comm());
    CHECK_EQ(got_gid_list.size(), gid_list.size());
    for (size_t k = 0; k < gid_list.size(); ++k) {
      CHECK_EQ(got_gid_list[k], gid_list[k]);
    }
  }
}

std::string random_string(size_t length) {
  const std::string characters =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<> distribution(0, characters.size() - 1);

  std::string ret;
  for (size_t i = 0; i < length; ++i) {
    ret += characters[distribution(generator)];
  }

  return ret;
}

std::string new_random_dir(const std::string& prefix, size_t length) {
  while (true) {
    std::string name = random_string(length);
    std::string path = prefix + "/" + name;
    int result = mkdir(path.c_str(), 0755);
    if (result == 0) {
      return path;
    }
  }
}

template <typename VM_T>
void TestVertexMap(const grape::CommSpec& comm_spec) {
  VM_T vm(comm_spec);
  vm.Init();
  using oid_t = typename VM_T::oid_t;
  std::vector<oid_t> id_list;
  LoadVertexMap(id_list, vm);

  grape::DistinctSort(id_list);

  ValidateVertexMap(id_list, vm);
  std::string prefix = new_random_dir(FLAGS_out_prefix, 8);
  vm.template Serialize<grape::LocalIOAdaptor>(prefix);

  VM_T new_vm(comm_spec);
  new_vm.Init();
  new_vm.template Deserialize<grape::LocalIOAdaptor>(prefix, comm_spec.fid());

  ValidateVertexMap(id_list, new_vm);
  rmdir(prefix.c_str());
}

int main(int argc, char** argv) {
  FLAGS_stderrthreshold = 0;

  grape::gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./run_app [grape_opts]");
  if (argc == 1) {
    grape::gflags::ShowUsageWithFlagsRestrict(argv[0], "analytical_apps");
    exit(1);
  }

  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  grape::gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("analytical_apps");
  google::InstallFailureSignalHandler();

  Init();

  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);

    TestVertexMap<grape::GlobalVertexMap<int64_t, uint32_t,
                                         grape::HashPartitioner<int64_t>>>(
        comm_spec);

    TestVertexMap<grape::GlobalVertexMap<int64_t, uint32_t,
                                         grape::SegmentedPartitioner<int64_t>>>(
        comm_spec);

    TestVertexMap<
        grape::GlobalVertexMap<int, uint32_t, grape::HashPartitioner<int>>>(
        comm_spec);

    TestVertexMap<grape::GlobalVertexMap<int, uint32_t,
                                         grape::SegmentedPartitioner<int>>>(
        comm_spec);

    TestVertexMap<grape::GlobalVertexMap<std::string, uint32_t,
                                         grape::HashPartitioner<std::string>>>(
        comm_spec);

    TestVertexMap<grape::ImmGlobalVertexMap<int64_t, uint32_t,
                                            grape::HashPartitioner<int64_t>>>(
        comm_spec);

    TestVertexMap<grape::ImmGlobalVertexMap<
        int64_t, uint32_t, grape::SegmentedPartitioner<int64_t>>>(comm_spec);

    TestVertexMap<grape::ImmGlobalVertexMap<int32_t, uint32_t,
                                            grape::HashPartitioner<int32_t>>>(
        comm_spec);

    TestVertexMap<grape::ImmGlobalVertexMap<
        int32_t, uint32_t, grape::SegmentedPartitioner<int32_t>>>(comm_spec);

    TestVertexMap<grape::ImmGlobalVertexMap<
        std::string, uint32_t, grape::HashPartitioner<std::string>>>(comm_spec);

    TestVertexMap<grape::PHGlobalVertexMap<int64_t, uint32_t,
                                           grape::HashPartitioner<int64_t>>>(
        comm_spec);

    TestVertexMap<grape::PHGlobalVertexMap<
        int64_t, uint32_t, grape::SegmentedPartitioner<int64_t>>>(comm_spec);

    TestVertexMap<
        grape::PHGlobalVertexMap<int, uint32_t, grape::HashPartitioner<int>>>(
        comm_spec);

    TestVertexMap<grape::PHGlobalVertexMap<int, uint32_t,
                                           grape::SegmentedPartitioner<int>>>(
        comm_spec);

    TestVertexMap<grape::PHGlobalVertexMap<
        std::string, uint32_t, grape::HashPartitioner<std::string>>>(comm_spec);
  }

  Finalize();

  google::ShutdownGoogleLogging();
}