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

#include "sssp/sssp.h"

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(out_prefix, "", "output directory of results");
DEFINE_int64(sssp_source, 0, "source vertex of sssp.");
DEFINE_string(loader_type, "basic", "loader type: basic, rb, efile or local");
DEFINE_bool(string_id, false, "whether to use string as origin id");

void Init() {
  if (FLAGS_out_prefix.empty()) {
    LOG(FATAL) << "Please assign an output prefix.";
  }
  if (FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input edge files.";
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

template <typename FRAG_T, typename APP_T, typename... Args>
void DoQuery(std::shared_ptr<FRAG_T> fragment, std::shared_ptr<APP_T> app,
             const grape::CommSpec& comm_spec,
             const grape::ParallelEngineSpec& spec,
             const std::string& out_prefix, Args... args) {
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  worker->Query(std::forward<Args>(args)...);

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  worker->Finalize();
}

template <typename T>
struct ParamConverter {};

template <>
struct ParamConverter<int64_t> {
  static int64_t FromInt64(int64_t val) { return val; }
};

template <>
struct ParamConverter<std::string> {
  static std::string FromInt64(int64_t val) { return std::to_string(val); }
};

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          grape::LoadStrategy load_strategy, template <class> class APP_T,
          typename... Args>
void CreateAndQuery(const grape::CommSpec& comm_spec,
                    const grape::LoadGraphSpec& graph_spec,
                    const std::string& out_prefix,
                    const grape::ParallelEngineSpec& spec, Args... args) {
  using FRAG_T = grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                 load_strategy>;
  std::shared_ptr<FRAG_T> fragment =
      grape::LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
  using AppType = APP_T<FRAG_T>;
  auto app = std::make_shared<AppType>();
  DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec, out_prefix,
                                    args...);
}

int main(int argc, char* argv[]) {
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
    grape::LoadGraphSpec graph_spec = grape::DefaultLoadGraphSpec();
    graph_spec.set_directed(false);
    if (FLAGS_loader_type == "rb") {
      graph_spec.set_rebalance(true, 0);
      graph_spec.partitioner_type = grape::PartitionerType::kMapPartitioner;
      // idxer_type = kMapIdxer;
    } else if (FLAGS_loader_type == "efile") {
      FLAGS_vfile = "";
      graph_spec.set_rebalance(false, 0);
      graph_spec.partitioner_type = grape::PartitionerType::kHashPartitioner;
      // idxer_type = kMapIdxer;
    } else if (FLAGS_loader_type == "local") {
      graph_spec.set_rebalance(false, 0);
      graph_spec.partitioner_type = grape::PartitionerType::kHashPartitioner;
      graph_spec.idxer_type = grape::IdxerType::kLocalIdxer;
    } else {
      CHECK_EQ(FLAGS_loader_type, "basic");
      graph_spec.set_rebalance(false, 0);

      // partitioner_type = kMapPartitioner;
      // idxer_type = kMapIdxer;
    }
    if (FLAGS_string_id) {
      CreateAndQuery<std::string, uint32_t, grape::EmptyType, double,
                     grape::LoadStrategy::kOnlyOut, grape::SSSP, std::string>(
          comm_spec, graph_spec, FLAGS_out_prefix,
          grape::DefaultParallelEngineSpec(),
          ParamConverter<std::string>::FromInt64(FLAGS_sssp_source));
    } else {
      CreateAndQuery<int64_t, uint32_t, grape::EmptyType, double,
                     grape::LoadStrategy::kOnlyOut, grape::SSSP, int64_t>(
          comm_spec, graph_spec, FLAGS_out_prefix,
          grape::DefaultParallelEngineSpec(),
          ParamConverter<int64_t>::FromInt64(FLAGS_sssp_source));
    }
  }
  Finalize();

  google::ShutdownGoogleLogging();
}
