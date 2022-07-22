#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_APP_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_APP_H_

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

#include "analytical_apps/app_config.h"
#include "analytical_apps/bfs/bfs.h"
#include "analytical_apps/bfs/bfs_fused.h"
#include "analytical_apps/cdlp/cdlp.h"
#include "analytical_apps/lcc/lcc.h"
#include "analytical_apps/mb/direct_access.h"
#include "analytical_apps/mb/serial_msg.h"
#include "analytical_apps/pagerank/async_delta_pagerank.h"
#include "analytical_apps/pagerank/async_pagerank.h"
#include "analytical_apps/pagerank/async_pagerank_ws.h"
#include "analytical_apps/pagerank/bc_pagerank.h"
#include "analytical_apps/pagerank/pagerank.h"
#include "analytical_apps/pagerank/pagerank_pull.h"
#include "analytical_apps/sssp/sssp.h"
#include "analytical_apps/sssp/sssp_bf.h"
#include "analytical_apps/sssp/sssp_fused.h"
#include "analytical_apps/sssp/sssp_nf.h"
#include "analytical_apps/sssp/sssp_old.h"
#include "analytical_apps/sssp/sssp_search.h"
#include "analytical_apps/sssp/sssp_stat.h"
#include "analytical_apps/sssp/sssp_ws.h"
#include "analytical_apps/wcc/wcc.h"
#include "analytical_apps/wcc/wcc_opt.h"
#include "flags.h"
#include "timer.h"
#include "grape/worker/comm_spec.h"
#include "grape_gpu/fragment/host_fragment.h"
#include "grape_gpu/fragment/loader.h"
#include "grape_gpu/fragment/random_partitioner.h"
#include "grape_gpu/fragment/remote_fragment.h"
#include "grape_gpu/worker/gpu_batch_shuffle_worker.h"
#include "grape_gpu/worker/gpu_worker.h"

namespace grape_gpu {

void Init() {
  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  grape::InitMPIComm();
}

void Finalize() {
  grape::FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

std::vector<int> parse_dev_list(const std::string& device) {
  std::string line;
  std::vector<int> vec;
  std::stringstream ss(device);
  char delim = ',';

  while (std::getline(ss, line, delim)) {
    vec.push_back(std::stoi(line));
  }

  return vec;
}

void set_device(const grape::CommSpec& comm_spec, const std::string& device) {
  auto vec = parse_dev_list(device);

  int dev = vec[comm_spec.local_id()];
  VLOG(1) << "Rank " << comm_spec.worker_id() << " is using device " << dev;
  CHECK_CUDA(cudaSetDevice(dev));
}

template <typename FRAG_T, typename APP_T, typename... Args>
void CreateAndQuery(const grape::CommSpec& comm_spec, const std::string& efile,
                    const std::string& vfile, const std::string& out_prefix,
                    Args... args) {
  using fragment_t = FRAG_T;
  using oid_t = typename FRAG_T::oid_t;

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;
  timer_start(is_coordinator);

  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_skip_first_valid_line(FLAGS_mtx);
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  graph_spec.set_rm_self_cycle(FLAGS_rm_self_cycle);
  graph_spec.set_kronec(FLAGS_scale, FLAGS_edgefactor);

  std::string partitioner = FLAGS_partitioner;
  std::string serialization_prefix = FLAGS_serialization_prefix;

  timer_next("load graph");
  std::shared_ptr<FRAG_T> fragment;

  auto dev_list = FLAGS_device;

  if (dev_list.empty()) {
    CHECK_CUDA(cudaSetDevice(comm_spec.local_id()));
  } else {
    set_device(comm_spec, dev_list);
  }

  if (partitioner == "seg") {
    fragment = LoadGraph<fragment_t, grape::SegmentedPartitioner<oid_t>>(
        efile, vfile, comm_spec, serialization_prefix, graph_spec);
  } else if (partitioner == "hash") {
    fragment = LoadGraph<fragment_t, grape::HashPartitioner<oid_t>>(
        efile, vfile, comm_spec, serialization_prefix, graph_spec);
  } else if (partitioner == "random") {
    fragment = LoadGraph<fragment_t, RandomPartitioner<oid_t>>(
        efile, vfile, comm_spec, serialization_prefix, graph_spec);
  } else if (partitioner == "seghash") {
    fragment = LoadGraph<fragment_t, SegHashPartitioner<oid_t>>(
        efile, vfile, comm_spec, serialization_prefix, graph_spec);
  }

  using app_t = APP_T;
  using worker_t = typename APP_T::worker_t;

  timer_next("load application");

  auto app = std::make_shared<app_t>();
  auto worker = std::make_shared<worker_t>(app, fragment);

  worker->Init(comm_spec, std::forward<Args>(args)...);

  // Wait all workers are built
  MPI_Barrier(comm_spec.comm());

  timer_next("run algorithm");

  worker->Query();
  // Wait all workers are finished
  MPI_Barrier(comm_spec.comm());
  timer_next("print output");

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, comm_spec.worker_id());
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
  }

  timer_end();
}

#define DEBUGGING
#undef DEBUGGING

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void Run() {
  grape::CommSpec comm_spec;

  comm_spec.Init(MPI_COMM_WORLD);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;

  if (FLAGS_debug) {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      sleep(1);
  }

  auto application = FLAGS_application;
  AppConfig app_config;

  app_config.lb = ParseLoadBalancing(FLAGS_lb);
  app_config.wl_alloc_factor_in = FLAGS_wl_alloc_factor_in;
  app_config.wl_alloc_factor_out_local = FLAGS_wl_alloc_factor_out_local;
  app_config.wl_alloc_factor_out_remote = FLAGS_wl_alloc_factor_out_remote;
  app_config.work_stealing = FLAGS_ws;
  app_config.ws_k = FLAGS_ws_k;

  if (application == "bfs") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = BFS<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_bfs_source);
#endif
  } else if (application == "bfs_fused") {
#ifndef DEBUGGING
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                     grape::LoadStrategy::kOnlyOut>;
    using AppType = BFSFused<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_bfs_source);
#endif
  } else if (application == "async_bfs") {
    //        using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
    //                                       grape::LoadStrategy::kOnlyOut>;
    //        using AppType = AsyncBFS<GraphType>;
    //        CreateAndQuery<GraphType, AppType>(
    //            comm_spec, efile, vfile, out_prefix, app_config,
    //            FLAGS_bfs_source, FLAGS_IPC_chunk_size, FLAGS_IPC_chunk_num,
    //            FLAGS_IPC_capacity);
  } else if (application == "sssp") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = SSSP<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source);
#endif
  } else if (application == "sssp_stat") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = SSSPStat<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source);
#endif
  } else if (application == "sssp_nf") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = SSSPNF<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source,
                                       FLAGS_sssp_prio);
#endif
  } else if (application == "sssp_old") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = SSSPOld<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source,
                                       FLAGS_sssp_prio);
#endif
  } else if (application == "sssp_ws") {
#ifndef DEBUGGING
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                     grape::LoadStrategy::kOnlyOut>;
    using AppType = SSSPWS<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source);
#endif
  } else if (application == "sssp_fused") {
    //#ifndef DEBUGGING
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                     grape::LoadStrategy::kOnlyOut>;
    using AppType = SSSPFused<GraphType>;
    CreateAndQuery<GraphType, AppType>(
        comm_spec, efile, vfile, out_prefix, app_config, FLAGS_sssp_source,
        FLAGS_sssp_prio, FLAGS_sssp_sw_round, parse_dev_list(FLAGS_mg_to));
    //#endif
  } else if (application == "sssp_bf") {
#ifdef FLOAT_WEIGHT
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, float,
                                     grape::LoadStrategy::kOnlyOut>;
#else
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                     grape::LoadStrategy::kOnlyOut>;
#endif
    using AppType = SSSPBF<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source,
                                       FLAGS_sssp_prio, FLAGS_sssp_mr);
  } else if (application == "sssp_search") {
#ifdef FLOAT_WEIGHT
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, float,
                                     grape::LoadStrategy::kOnlyOut>;
#else
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                     grape::LoadStrategy::kOnlyOut>;
#endif
    using AppType = SSSPSEARCH<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_sssp_source,
                                       FLAGS_sssp_prio, FLAGS_sssp_mr);
  } else if (application == "wcc") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = WCC<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config);
#endif
  } else if (application == "wcc_opt") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = WCCOpt<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config);
#endif
  } else if (application == "pagerank") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = Pagerank<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d, FLAGS_pr_mr);
#endif
  } else if (application == "async_pagerank_ws") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = AsyncPagerankWS<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d,
                                       FLAGS_apr_epslion);
#endif
  } else if (application == "async_pagerank") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = AsyncDeltaPagerank<GraphType>;
    CreateAndQuery<GraphType, AppType>(
        comm_spec, efile, vfile, out_prefix, app_config, FLAGS_pr_d,
        FLAGS_apr_async, FLAGS_apr_breakdown, FLAGS_apr_endure, FLAGS_apr_mc);
#endif
  } else if (application == "async_pagerank1") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = AsyncPagerank<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d,
                                       FLAGS_apr_breakdown, FLAGS_apr_mc);
#endif
  } else if (application == "bs_pagerank") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = BCPagerank<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d,
                                       FLAGS_apr_breakdown);
#endif
  } else if (application == "pagerank_pull") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kBothOutIn>;
    using AppType = PagerankPull<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_pr_d, FLAGS_pr_mr);
#endif
  } else if (application == "lcc") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = LCC<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config);
#endif
  } else if (application == "cdlp") {
#ifndef DEBUGGING
    using GraphType = HostFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                   grape::LoadStrategy::kOnlyOut>;
    using AppType = CDLP<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_cdlp_mr);
#endif
  } else if (application == "mb_direct_access") {
#ifndef DEBUGGING
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                     grape::LoadStrategy::kOnlyOut>;
    using AppType = MBDirectAccess<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_mb_steal_factor);
#endif
  } else if (application == "mb_serial_msg") {
#ifndef DEBUGGING
    using GraphType = RemoteFragment<OID_T, VID_T, VDATA_T, uint32_t,
                                     grape::LoadStrategy::kOnlyOut>;
    using AppType = MBSerialMsg<GraphType>;
    CreateAndQuery<GraphType, AppType>(comm_spec, efile, vfile, out_prefix,
                                       app_config, FLAGS_mb_steal_factor);
#endif
  } else {
    LOG(FATAL) << "Invalid app name: " << application;
  }
}
}  // namespace grape_gpu
#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_APP_H_
