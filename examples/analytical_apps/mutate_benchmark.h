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

#ifndef EXAMPLES_ANALYTICAL_APPS_MUTATE_BENCHMARK_H_
#define EXAMPLES_ANALYTICAL_APPS_MUTATE_BENCHMARK_H_

#include <sys/stat.h>

#include <grape/fragment/ev_fragment_mutator.h>
#include <grape/fragment/immutable_edgecut_fragment.h>
#include <grape/fragment/loader.h>
#include <grape/fragment/mutable_edgecut_fragment.h>
#include <grape/grape.h>
#include <grape/util.h>

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include "mutate_benchmark_flags.h"
#include "pagerank/pagerank_local_parallel.h"
#include "sssp/sssp.h"
#include "timer.h"

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

namespace grape {

void Init() {
  if (FLAGS_out_prefix.empty()) {
    LOG(FATAL) << "Please assign an output prefix.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  InitMPIComm();
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  if (comm_spec.worker_id() == kCoordinatorRank) {
    VLOG(1) << "Workers of libgrape-lite initialized.";
  }
}

void Finalize() {
  FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T>
std::shared_ptr<FRAG_T> BuildGraph(const CommSpec& comm_spec,
                                   const std::string& efile,
                                   const std::string& vfile) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(false, 0);
  graph_spec.set_deserialize(false, "");
  graph_spec.set_serialize(false, "");
  std::shared_ptr<FRAG_T> fragment;
  fragment = LoadGraph<FRAG_T>(efile, vfile, comm_spec, graph_spec);
  return fragment;
}

template <typename FRAG_T>
std::shared_ptr<FRAG_T> MutateGraph(const CommSpec& comm_spec,
                                    const std::string& efile_prefix,
                                    int efile_num,
                                    std::shared_ptr<FRAG_T> fragment) {
  timer_next("mutate graph");
  EVFragmentMutator<FRAG_T, LocalIOAdaptor> mutator(comm_spec);
  for (int i = 0; i < efile_num; ++i) {
    std::string path = efile_prefix + ".part_" + std::to_string(i);
    fragment = mutator.MutateFragment(path, "", fragment, FLAGS_directed);
  }
  return fragment;
}

template <typename FRAG_T, typename APP_T, typename... Args>
void RunQuery(std::shared_ptr<FRAG_T> fragment, const CommSpec& comm_spec,
              const std::string& out_prefix, const ParallelEngineSpec& spec,
              Args... args) {
  timer_next("load application");
  auto app = std::make_shared<APP_T>();
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  timer_next("run algorithm");
  worker->Query(std::forward<Args>(args)...);
  timer_next("print output");
  std::ofstream ostream;
  std::string output_path = GetResultFilename(out_prefix, fragment->fid());
  ostream.open(output_path);
  worker->Output(ostream);
  ostream.close();
  worker->Finalize();
  timer_end();
  VLOG(1) << "Worker-" << comm_spec.worker_id() << " finished: " << output_path;
}

template <typename FRAG_T, typename APP_T, typename... Args>
void BuildGraphAndQuery(const CommSpec& comm_spec, const std::string& efile,
                        const std::string& vfile,
                        const std::string& efile_prefix, int efile_num,
                        const std::string& out_prefix,
                        const ParallelEngineSpec& spec, Args... args) {
  std::shared_ptr<FRAG_T> fragment =
      BuildGraph<FRAG_T>(comm_spec, efile, vfile);
  fragment = MutateGraph(comm_spec, efile_prefix, efile_num, fragment);
  RunQuery<FRAG_T, APP_T, Args...>(fragment, comm_spec, out_prefix, spec,
                                   std::forward<Args>(args)...);
}

template <typename FRAG_T, typename APP_T, typename... Args>
void BuildImmutableGraphAndQuery(const CommSpec& comm_spec,
                                 const std::string& efile,
                                 const std::string& vfile,
                                 const std::string& out_prefix,
                                 const ParallelEngineSpec& spec, Args... args) {
  std::shared_ptr<FRAG_T> fragment =
      BuildGraph<FRAG_T>(comm_spec, efile, vfile);
  RunQuery<FRAG_T, APP_T, Args...>(fragment, comm_spec, out_prefix, spec,
                                   std::forward<Args>(args)...);
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void RunBenchmark() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string delta_efile_prefix = FLAGS_delta_efile_prefix;
  int delta_efile_part_num = FLAGS_delta_efile_part_num;
  std::string out_prefix = FLAGS_out_prefix;

  auto spec = MultiProcessSpec(comm_spec, __AFFINITY__);

  if (delta_efile_part_num != 0) {
    using GraphType = MutableEdgecutFragment<OID_T, VID_T, VDATA_T, double>;
    using AppType = SSSP<GraphType>;
    BuildGraphAndQuery<GraphType, AppType, OID_T>(
        comm_spec, efile, vfile, delta_efile_prefix, delta_efile_part_num,
        out_prefix, spec, FLAGS_sssp_source);
  } else {
    using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, double>;
    using AppType = SSSP<GraphType>;
    BuildImmutableGraphAndQuery<GraphType, AppType, OID_T>(
        comm_spec, efile, vfile, out_prefix, spec, FLAGS_sssp_source);
  }
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void RunPageRankBenchmark() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);
  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string delta_efile_prefix = FLAGS_delta_efile_prefix;
  int delta_efile_part_num = FLAGS_delta_efile_part_num;
  std::string out_prefix = FLAGS_out_prefix;

  auto spec = MultiProcessSpec(comm_spec, __AFFINITY__);

  if (delta_efile_part_num != 0) {
    using GraphType = MutableEdgecutFragment<OID_T, VID_T, VDATA_T, double,
                                             LoadStrategy::kBothOutIn>;
    using AppType = PageRankLocalParallel<GraphType>;
    BuildGraphAndQuery<GraphType, AppType, double, int>(
        comm_spec, efile, vfile, delta_efile_prefix, delta_efile_part_num,
        out_prefix, spec, FLAGS_pr_d, FLAGS_pr_mr);
  } else {
    using GraphType = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, double,
                                               LoadStrategy::kBothOutIn>;
    using AppType = PageRankLocalParallel<GraphType>;
    BuildImmutableGraphAndQuery<GraphType, AppType, double, int>(
        comm_spec, efile, vfile, out_prefix, spec, FLAGS_pr_d, FLAGS_pr_mr);
  }
}

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_MUTATE_BENCHMARK_H_
