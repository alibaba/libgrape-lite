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

#include <sys/stat.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <grape/grape.h>
#include <grape/util.h>

#include "cuda/bfs/bfs.h"
#include "cuda/cdlp/cdlp.h"
#include "cuda/lcc/lcc.h"
#include "cuda/lcc/lcc_directed.h"
#include "cuda/lcc/lcc_directed_opt.h"
#include "cuda/lcc/lcc_directed_preprocess.h"
#include "cuda/lcc/lcc_opt.h"
#include "cuda/lcc/lcc_preprocess.h"
#include "cuda/pagerank/pagerank.h"
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

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

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
void DoPreprocess(std::shared_ptr<FRAG_T> fragment, std::shared_ptr<APP_T> app,
                  const CommSpec& comm_spec, int dev_id,
                  const std::string& out_prefix, Args... args) {
  auto spec = MultiProcessSpec(comm_spec, __AFFINITY__);
  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
    if (__AFFINITY__) {
      if (spec.cpu_list.size() >= spec.thread_num) {
        spec.cpu_list.resize(spec.thread_num);
      } else {
        uint32_t num_to_append = spec.thread_num - spec.cpu_list.size();
        for (uint32_t i = 0; i < num_to_append; ++i) {
          spec.cpu_list.push_back(spec.cpu_list[i]);
        }
      }
    }
  }
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  MPI_Barrier(comm_spec.comm());
  worker->Query(std::forward<Args>(args)...);
  MPI_Barrier(comm_spec.comm());
  std::ofstream ostream;
  worker->Output(ostream);
  worker->Finalize();
}

template <typename FRAG_T, typename APP_T, typename... Args>
void DoQuery(std::shared_ptr<FRAG_T> fragment, std::shared_ptr<APP_T> app,
             const CommSpec& comm_spec, int dev_id,
             const std::string& out_prefix, Args... args) {
  timer_next("load application");
  auto worker = APP_T::CreateWorker(app, fragment);
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
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker->Finalize();
  timer_end();
}
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy load_strategy, template <class> class APP_T,
          template <class> class PRE_T, typename... Args>
void CreateAndQueryWithPreprocess(const grape::CommSpec& comm_spec,
                                  const std::string& efile,
                                  const std::string& vfile,
                                  const std::string& out_prefix, Args... args) {
  // using fragment_t = FRAG_T;
  // using oid_t = typename FRAG_T::oid_t;
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  if (FLAGS_segmented_partition) {
    using VERTEX_MAP_T =
        GlobalVertexMap<OID_T, VID_T, SegmentedPartitioner<OID_T>>;
    using FRAG_T = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                             load_strategy, VERTEX_MAP_T>;
    std::shared_ptr<FRAG_T> fragment;
    int dev_id = comm_spec.local_id();
    int dev_count;

    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    CHECK_LE(comm_spec.local_num(), dev_count)
        << "Only found " << dev_count << " GPUs, but " << comm_spec.local_num()
        << " processes are launched";
    CHECK_CUDA(cudaSetDevice(dev_id));
    fragment = LoadGraph<FRAG_T>(efile, vfile, comm_spec, graph_spec);

    auto app = std::make_shared<APP_T<FRAG_T>>();
    auto pre = std::make_shared<PRE_T<FRAG_T>>();
    DoPreprocess<FRAG_T, PRE_T<FRAG_T>, Args...>(fragment, pre, comm_spec,
                                                 dev_id, out_prefix, args...);
    DoQuery<FRAG_T, APP_T<FRAG_T>, Args...>(fragment, app, comm_spec, dev_id,
                                            out_prefix, args...);
  } else {
    graph_spec.set_rebalance(false, 0);
    using VERTEX_MAP_T = GlobalVertexMap<OID_T, VID_T, HashPartitioner<OID_T>>;
    using FRAG_T = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                             load_strategy, VERTEX_MAP_T>;
    std::shared_ptr<FRAG_T> fragment;
    int dev_id = comm_spec.local_id();
    int dev_count;

    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    CHECK_LE(comm_spec.local_num(), dev_count)
        << "Only found " << dev_count << " GPUs, but " << comm_spec.local_num()
        << " processes are launched";
    CHECK_CUDA(cudaSetDevice(dev_id));
    fragment = LoadGraph<FRAG_T>(efile, vfile, comm_spec, graph_spec);

    auto app = std::make_shared<APP_T<FRAG_T>>();
    auto pre = std::make_shared<PRE_T<FRAG_T>>();
    DoPreprocess<FRAG_T, PRE_T<FRAG_T>, Args...>(fragment, pre, comm_spec,
                                                 dev_id, out_prefix, args...);
    DoQuery<FRAG_T, APP_T<FRAG_T>, Args...>(fragment, app, comm_spec, dev_id,
                                            out_prefix, args...);
  }
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy load_strategy, template <class> class APP_T,
          typename... Args>
void CreateAndQuery(const grape::CommSpec& comm_spec, const std::string& efile,
                    const std::string& vfile, const std::string& out_prefix,
                    Args... args) {
  // using fragment_t = FRAG_T;
  // using oid_t = typename FRAG_T::oid_t;
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  if (FLAGS_segmented_partition) {
    using VERTEX_MAP_T =
        GlobalVertexMap<OID_T, VID_T, SegmentedPartitioner<OID_T>>;
    using FRAG_T = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                             load_strategy, VERTEX_MAP_T>;
    std::shared_ptr<FRAG_T> fragment;
    int dev_id = comm_spec.local_id();
    int dev_count;

    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    CHECK_LE(comm_spec.local_num(), dev_count)
        << "Only found " << dev_count << " GPUs, but " << comm_spec.local_num()
        << " processes are launched";
    CHECK_CUDA(cudaSetDevice(dev_id));
    fragment = LoadGraph<FRAG_T>(efile, vfile, comm_spec, graph_spec);

    auto app = std::make_shared<APP_T<FRAG_T>>();
    DoQuery<FRAG_T, APP_T<FRAG_T>, Args...>(fragment, app, comm_spec, dev_id,
                                            out_prefix, args...);
  } else {
    graph_spec.set_rebalance(false, 0);
    using VERTEX_MAP_T = GlobalVertexMap<OID_T, VID_T, HashPartitioner<OID_T>>;
    using FRAG_T = grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                             load_strategy, VERTEX_MAP_T>;
    std::shared_ptr<FRAG_T> fragment;
    int dev_id = comm_spec.local_id();
    int dev_count;

    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    CHECK_LE(comm_spec.local_num(), dev_count)
        << "Only found " << dev_count << " GPUs, but " << comm_spec.local_num()
        << " processes are launched";
    CHECK_CUDA(cudaSetDevice(dev_id));
    fragment = LoadGraph<FRAG_T>(efile, vfile, comm_spec, graph_spec);

    auto app = std::make_shared<APP_T<FRAG_T>>();
    DoQuery<FRAG_T, APP_T<FRAG_T>, Args...>(fragment, app, comm_spec, dev_id,
                                            out_prefix, args...);
  }
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
  app_config.wl_alloc_factor_in = 0.4;
  app_config.wl_alloc_factor_out_local = 0.2;
  app_config.wl_alloc_factor_out_remote = 0.2;

  if (application == "bfs") {
    if (FLAGS_directed) {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kBothOutIn, BFS>(
          comm_spec, efile, vfile, out_prefix, app_config, FLAGS_bfs_source);
    } else {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kOnlyOut, BFS>(
          comm_spec, efile, vfile, out_prefix, app_config, FLAGS_bfs_source);
    }
  } else if (application == "sssp") {
#ifdef INT_WEIGHT
    using WeightT = uint32_t;
#else
    using WeightT = float;
#endif
    CreateAndQuery<OID_T, VID_T, VDATA_T, WeightT,
                   grape::LoadStrategy::kOnlyOut, SSSP>(
        comm_spec, efile, vfile, out_prefix, app_config, FLAGS_sssp_source, 0);
  } else if (application == "wcc") {
    if (FLAGS_directed) {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kBothOutIn, WCC>(
          comm_spec, efile, vfile, out_prefix, app_config);
    } else {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kOnlyOut, WCC>(
          comm_spec, efile, vfile, out_prefix, app_config);
    }
  } else if (application == "wcc_opt") {
    CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                   grape::LoadStrategy::kOnlyOut, WCCOpt>(
        comm_spec, efile, vfile, out_prefix, app_config);
  } else if (application == "pagerank") {
    if (FLAGS_directed) {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kBothOutIn, Pagerank>(
          comm_spec, efile, vfile, out_prefix, app_config, FLAGS_pr_d,
          FLAGS_pr_mr);
    } else {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kOnlyOut, Pagerank>(
          comm_spec, efile, vfile, out_prefix, app_config, FLAGS_pr_d,
          FLAGS_pr_mr);
    }
  } else if (application == "lcc") {
    VID_T** col = reinterpret_cast<VID_T**>(malloc(sizeof(VID_T*)));
    size_t** row_offset = reinterpret_cast<size_t**>(malloc(sizeof(size_t*)));
    if (FLAGS_directed) {
      char** weight = reinterpret_cast<char**>(malloc(sizeof(char*)));
      size_t** true_degree =
          reinterpret_cast<size_t**>(malloc(sizeof(size_t*)));
      CreateAndQueryWithPreprocess<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kBothOutIn, LCCDOPT,
                                   LCCDP>(comm_spec, efile, vfile, out_prefix,
                                          app_config, col, row_offset, weight,
                                          true_degree);
    } else {
      CreateAndQueryWithPreprocess<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut, LCCOPT, LCCP>(
          comm_spec, efile, vfile, out_prefix, app_config, col, row_offset);
    }
  } else if (application == "cdlp") {
    if (FLAGS_directed) {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kBothOutIn, CDLP>(
          comm_spec, efile, vfile, out_prefix, app_config, FLAGS_cdlp_mr);
    } else {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EDATA_T,
                     grape::LoadStrategy::kOnlyOut, CDLP>(
          comm_spec, efile, vfile, out_prefix, app_config, FLAGS_cdlp_mr);
    }
  } else {
    LOG(FATAL) << "Invalid app name: " << application;
  }
}
}  // namespace cuda
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_
