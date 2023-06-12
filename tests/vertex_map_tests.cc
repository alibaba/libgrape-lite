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
#include <grape/fragment/partitioner.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <grape/vertex_map/global_vertex_map.h>
#include <grape/vertex_map/local_vertex_map.h>

#include "sssp/sssp.h"
#include "timer.h"

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(delta_efile, "", "delta edge file");
DEFINE_string(delta_vfile, "", "delta vertex file");
DEFINE_string(out_prefix, "", "output directory of results");
DEFINE_int64(sssp_source, 0, "source vertex of sssp.");
DEFINE_bool(string_id, false, "whether to use string as origin id");
DEFINE_bool(segmented_partition, true,
            "whether to use segmented partitioning.");
DEFINE_bool(rebalance, false, "whether to rebalance graph after loading.");
DEFINE_int32(rebalance_vertex_factor, 0, "vertex factor of rebalancing.");
DEFINE_bool(global_vertex_map, true, "whether to use global vertex map.");

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
  timer_next("load application");
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  timer_next("run algorithm");
  worker->Query(std::forward<Args>(args)...);
  timer_next("print output");

  std::ofstream ostream;
  std::string output_path =
      grape::GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  worker->Finalize();
  timer_end();
  VLOG(1) << "Worker-" << comm_spec.worker_id() << " finished: " << output_path;
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
                    const std::string& out_prefix, int fnum,
                    const grape::ParallelEngineSpec& spec, Args... args) {
  timer_next("load graph");
  grape::LoadGraphSpec graph_spec = grape::DefaultLoadGraphSpec();
  graph_spec.set_directed(false);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (!FLAGS_delta_efile.empty() || !FLAGS_delta_vfile.empty()) {
    graph_spec.set_rebalance(false, 0);
    if (FLAGS_global_vertex_map) {
      using VertexMapType = grape::GlobalVertexMap<OID_T, VID_T>;
      using FRAG_T =
          grape::MutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                        load_strategy, VertexMapType>;
      std::shared_ptr<FRAG_T> fragment = grape::LoadGraphAndMutate<FRAG_T>(
          FLAGS_efile, FLAGS_vfile, FLAGS_delta_efile, FLAGS_delta_vfile,
          comm_spec, graph_spec);
      using AppType = APP_T<FRAG_T>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                        out_prefix, args...);
    } else {
      using VertexMapType = grape::LocalVertexMap<OID_T, VID_T>;
      using FRAG_T =
          grape::MutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                        load_strategy, VertexMapType>;
      std::shared_ptr<FRAG_T> fragment = grape::LoadGraphAndMutate<FRAG_T>(
          FLAGS_efile, FLAGS_vfile, FLAGS_delta_efile, FLAGS_delta_vfile,
          comm_spec, graph_spec);
      using AppType = APP_T<FRAG_T>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                        out_prefix, args...);
    }
  } else {
    if (FLAGS_segmented_partition) {
      if (FLAGS_global_vertex_map) {
        using VertexMapType =
            grape::GlobalVertexMap<OID_T, VID_T,
                                   grape::SegmentedPartitioner<OID_T>>;
        using FRAG_T =
            grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                            load_strategy, VertexMapType>;
        std::shared_ptr<FRAG_T> fragment = grape::LoadGraph<FRAG_T>(
            FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
        using AppType = APP_T<FRAG_T>;
        auto app = std::make_shared<AppType>();
        DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                          out_prefix, args...);
      } else {
        using VertexMapType =
            grape::LocalVertexMap<OID_T, VID_T,
                                  grape::SegmentedPartitioner<OID_T>>;
        using FRAG_T =
            grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                            load_strategy, VertexMapType>;
        std::shared_ptr<FRAG_T> fragment = grape::LoadGraph<FRAG_T>(
            FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
        using AppType = APP_T<FRAG_T>;
        auto app = std::make_shared<AppType>();
        DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                          out_prefix, args...);
      }
    } else {
      graph_spec.set_rebalance(false, 0);
      if (FLAGS_global_vertex_map) {
        using VertexMapType = grape::GlobalVertexMap<OID_T, VID_T>;
        using FRAG_T =
            grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                            load_strategy, VertexMapType>;
        std::shared_ptr<FRAG_T> fragment = grape::LoadGraph<FRAG_T>(
            FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
        using AppType = APP_T<FRAG_T>;
        auto app = std::make_shared<AppType>();
        DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                          out_prefix, args...);
      } else {
        using VertexMapType = grape::LocalVertexMap<OID_T, VID_T>;
        using FRAG_T =
            grape::ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                            load_strategy, VertexMapType>;
        std::shared_ptr<FRAG_T> fragment = grape::LoadGraph<FRAG_T>(
            FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
        using AppType = APP_T<FRAG_T>;
        auto app = std::make_shared<AppType>();
        DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                          out_prefix, args...);
      }
    }
  }
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void Run() {
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;
  timer_start(is_coordinator);

  // FIXME: no barrier apps. more manager? or use a dynamic-cast.
  auto spec = grape::MultiProcessSpec(comm_spec, __AFFINITY__);
  int fnum = comm_spec.fnum();
  CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T, grape::LoadStrategy::kOnlyOut,
                 grape::SSSP, OID_T>(
      comm_spec, FLAGS_out_prefix, fnum, spec,
      ParamConverter<OID_T>::FromInt64(FLAGS_sssp_source));
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

  if (FLAGS_string_id) {
    Run<std::string, uint32_t, grape::EmptyType, double>();
  } else {
    Run<int64_t, uint32_t, grape::EmptyType, double>();
  }

  Finalize();

  google::ShutdownGoogleLogging();
}
