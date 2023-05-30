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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <grape/fragment/loader.h>
#include <grape/fragment/mutable_edgecut_fragment.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <grape/vertex_map/global_vertex_map.h>

#include "bfs/bfs.h"
#include "bfs/bfs_auto.h"
#include "cdlp/cdlp.h"
#include "cdlp/cdlp_auto.h"
#include "flags.h"
#include "lcc/lcc.h"
#include "lcc/lcc_auto.h"
#include "pagerank/pagerank.h"
#include "pagerank/pagerank_auto.h"
#include "pagerank/pagerank_local.h"
#include "pagerank/pagerank_local_parallel.h"
#include "pagerank/pagerank_parallel.h"
#include "sssp/sssp.h"
#include "sssp/sssp_auto.h"
#include "timer.h"
#include "wcc/wcc.h"
#include "wcc/wcc_auto.h"

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

DEFINE_string(efile, "", "edge file");
DEFINE_string(vfile, "", "vertex file");
DEFINE_string(delta_efile, "", "delta edge file");
DEFINE_string(delta_vfile, "", "delta vertex file");
DEFINE_string(out_prefix, "", "output directory of results");
DEFINE_int64(bfs_source, 0, "source vertex of bfs.");
DEFINE_int32(cdlp_mr, 10, "max rounds of cdlp.");
DEFINE_int64(sssp_source, 0, "source vertex of sssp.");
DEFINE_double(pr_d, 0.85, "damping_factor of pagerank");
DEFINE_int32(pr_mr, 10, "max rounds of pagerank");
DEFINE_bool(directed, false, "input graph is directed or not.");
DEFINE_string(application, "", "application name");

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
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  using FRAG_T = grape::MutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                               load_strategy>;
  std::shared_ptr<FRAG_T> fragment = grape::LoadGraphAndMutate<FRAG_T>(
      FLAGS_efile, FLAGS_vfile, FLAGS_delta_efile, FLAGS_delta_vfile, comm_spec,
      graph_spec);
  using AppType = APP_T<FRAG_T>;
  auto app = std::make_shared<AppType>();
  DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec, out_prefix,
                                    args...);
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void Run() {
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;
  timer_start(is_coordinator);

  // FIXME: no barrier apps. more manager? or use a dynamic-cast.
  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string delta_efile = FLAGS_delta_efile;
  std::string delta_vfile = FLAGS_delta_vfile;
  std::string out_prefix = FLAGS_out_prefix;
  auto spec = grape::MultiProcessSpec(comm_spec, __AFFINITY__);
  int fnum = comm_spec.fnum();
  std::string name = FLAGS_application;
  if (name.find("sssp") != std::string::npos) {
    if (name == "sssp") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, double,
                     grape::LoadStrategy::kOnlyOut, grape::SSSP, OID_T>(
          comm_spec, out_prefix, fnum, spec,
          ParamConverter<OID_T>::FromInt64(FLAGS_sssp_source));
    } else if (name == "sssp_auto") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, double,
                     grape::LoadStrategy::kOnlyOut, grape::SSSPAuto, OID_T>(
          comm_spec, out_prefix, fnum, spec,
          ParamConverter<OID_T>::FromInt64(FLAGS_sssp_source));
    } else {
      LOG(FATAL) << "No avaiable application named [" << name << "].";
    }
  } else {
    if (name == "bfs") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::BFS, OID_T>(
          comm_spec, out_prefix, fnum, spec,
          ParamConverter<OID_T>::FromInt64(FLAGS_bfs_source));
    } else if (name == "bfs_auto") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::BFSAuto, OID_T>(
          comm_spec, out_prefix, fnum, spec,
          ParamConverter<OID_T>::FromInt64(FLAGS_bfs_source));
    } else if (name == "pagerank_local") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::PageRankLocal,
                     double, int>(comm_spec, out_prefix, fnum, spec, FLAGS_pr_d,
                                  FLAGS_pr_mr);
    } else if (name == "pagerank_local_parallel") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kBothOutIn,
                     grape::PageRankLocalParallel, double, int>(
          comm_spec, out_prefix, fnum, spec, FLAGS_pr_d, FLAGS_pr_mr);
    } else if (name == "pagerank") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::PageRank, double,
                     int>(comm_spec, out_prefix, fnum, spec, FLAGS_pr_d,
                          FLAGS_pr_mr);
    } else if (name == "pagerank_auto") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kBothOutIn, grape::PageRankAuto,
                     double, int>(comm_spec, out_prefix, fnum, spec, FLAGS_pr_d,
                                  FLAGS_pr_mr);
    } else if (name == "pagerank_parallel") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kBothOutIn, grape::PageRankParallel,
                     double, int>(comm_spec, out_prefix, fnum, spec, FLAGS_pr_d,
                                  FLAGS_pr_mr);
    } else if (name == "cdlp") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::CDLP, int>(
          comm_spec, out_prefix, fnum, spec, FLAGS_cdlp_mr);
    } else if (name == "cdlp_auto") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kBothOutIn, grape::CDLPAuto, int>(
          comm_spec, out_prefix, fnum, spec, FLAGS_cdlp_mr);
    } else if (name == "wcc") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::WCC>(
          comm_spec, out_prefix, fnum, spec);
    } else if (name == "wcc_auto") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::WCCAuto>(
          comm_spec, out_prefix, fnum, spec);
    } else if (name == "lcc") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::LCC>(
          comm_spec, out_prefix, fnum, spec);
    } else if (name == "lcc_auto") {
      CreateAndQuery<OID_T, VID_T, VDATA_T, grape::EmptyType,
                     grape::LoadStrategy::kOnlyOut, grape::LCCAuto>(
          comm_spec, out_prefix, fnum, spec);
    } else {
      LOG(FATAL) << "No avaiable application named [" << name << "].";
    }
  }
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

  std::string name = FLAGS_application;
  if (name.find("sssp") != std::string::npos) {
    Run<int64_t, uint32_t, grape::EmptyType, double>();
  } else {
    Run<int64_t, uint32_t, grape::EmptyType, grape::EmptyType>();
  }

  Finalize();

  google::ShutdownGoogleLogging();
}
