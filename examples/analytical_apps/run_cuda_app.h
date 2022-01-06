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

#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda/bfs/bfs.h"
#include "cuda/cdlp/cdlp.h"
#include "cuda/lcc/lcc.h"
#include "cuda/pagerank/pagerank.h"
#include "cuda/pagerank/pagerank_pull.h"
#include "cuda/sssp/sssp.h"
#include "cuda/wcc/wcc.h"
#include "cuda/wcc/wcc_opt.h"

#include "flags.h"
#include "grape/cuda/fragment/host_fragment.h"
#include "grape/cuda/worker/gpu_batch_shuffle_worker.h"
#include "grape/cuda/worker/gpu_worker.h"
#include "grape/fragment/loader.h"
#include "grape/worker/comm_spec.h"
#include "timer.h"

namespace grape {

namespace cuda {

void Init() {
  if (FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  grape::InitMPIComm();
}

void Finalize() {
  grape::FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const grape::CommSpec& comm_spec, const std::string& efile,
                    const std::string& vfile, const std::string& out_prefix,
                    Args... args) {
  // using fragment_t = FRAG_T;
  // using oid_t = typename FRAG_T::oid_t;
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  graph_spec.serialization_prefix = FLAGS_serialization_prefix;

  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  std::shared_ptr<FRAG_T> fragment;
  int dev_id = comm_spec.local_id();
  int dev_count;

  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  CHECK_LE(comm_spec.local_num(), dev_count)
      << "Only found " << dev_count << " GPUs, but " << comm_spec.local_num()
      << " processes are launched";
  CHECK_CUDA(cudaSetDevice(dev_id));

  if (FLAGS_segmented_partition) {
    fragment = LoadGraph<FRAG_T, SegmentedPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  } else {
    fragment = LoadGraph<FRAG_T, HashPartitioner<typename FRAG_T::oid_t>>(
        efile, vfile, comm_spec, graph_spec);
  }

  auto app = std::make_shared<APP_T>();
  timer_next("load application");
  auto worker = APP_T::CreateWorker(app, fragment);  // FIXME
  worker->Init(comm_spec, std::forward<Args>(args)...);
  MPI_Barrier(comm_spec.comm());

  timer_next("run algorithm");
  CHECK_CUDA(cudaSetDevice(dev_id));
  worker->Query();
  MPI_Barrier(comm_spec.comm());

  timer_next("print output");

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id() << " finished: " << output_path;
  }
  worker->Finalize();
  timer_end();
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void Run() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;
  timer_start(is_coordinator);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;

  auto application = FLAGS_application;
  AppConfig app_config;

  app_config.lb = ParseLoadBalancing(FLAGS_lb);
  app_config.wl_alloc_factor_in = FLAGS_wl_alloc_factor_in;
  app_config.wl_alloc_factor_out_local = FLAGS_wl_alloc_factor_out_local;
  app_config.wl_alloc_factor_out_remote = FLAGS_wl_alloc_factor_out_remote;

  if (application == "bfs") {
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                grape::LoadStrategy::kOnlyOut>;
    using AppType = BFS<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_bfs_source);
  } else if (application == "sssp") {
#ifdef FLOAT_WEIGHT
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, float,
                                                grape::LoadStrategy::kOnlyOut>;
#else
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                                grape::LoadStrategy::kOnlyOut>;
#endif
    using AppType = SSSP<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source,
                                       FLAGS_sssp_prio);
  } else if (application == "wcc") {
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                grape::LoadStrategy::kOnlyOut>;
    using AppType = WCC<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config);
  } else if (application == "wcc_opt") {
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                grape::LoadStrategy::kOnlyOut>;
    using AppType = WCCOpt<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config);
  } else if (application == "pagerank") {
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                grape::LoadStrategy::kOnlyOut>;
    using AppType = Pagerank<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d, FLAGS_pr_mr);
  } else if (application == "pagerank_pull") {
    using GraphType =
        grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                  grape::LoadStrategy::kBothOutIn>;
    using AppType = PagerankPull<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d, FLAGS_pr_mr);
  } else if (application == "lcc") {
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                grape::LoadStrategy::kOnlyOut>;
    using AppType = LCC<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config);
  } else if (application == "cdlp") {
    using GraphType = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                                grape::LoadStrategy::kOnlyOut>;
    using AppType = CDLP<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_cdlp_mr);
  } else {
    LOG(FATAL) << "Invalid app name: " << application;
  }
}
}  // namespace cuda
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_
