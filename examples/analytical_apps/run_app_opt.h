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

#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_APP_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_APP_OPT_H_

#include "bfs/bfs_opt.h"
#include "cdlp/cdlp_opt.h"
#include "cdlp/cdlp_opt_ud.h"
#include "cdlp/cdlp_opt_ud_dense.h"
#include "lcc/lcc_beta.h"
#include "lcc/lcc_directed.h"
#include "lcc/lcc_opt.h"
#include "pagerank/pagerank_directed.h"
#include "pagerank/pagerank_opt.h"
#include "pagerank/pagerank_push_opt.h"
#include "run_app.h"
#include "sssp/sssp_opt.h"
#include "wcc/wcc_opt.h"

namespace grape {

template <typename FRAG_T>
using LCC64 = LCCOpt<FRAG_T, uint64_t>;

template <typename FRAG_T>
using LCCBeta64 = LCCBeta<FRAG_T, uint64_t>;

template <typename FRAG_T>
using LCCDirected64 = LCCDirected<FRAG_T, uint64_t>;

template <typename FRAG_T>
using LCC32 = LCCOpt<FRAG_T, uint32_t>;

template <typename FRAG_T>
using LCCBeta32 = LCCBeta<FRAG_T, uint32_t>;

template <typename FRAG_T>
using LCCDirected32 = LCCDirected<FRAG_T, uint32_t>;

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy load_strategy>
void RunUndirectedPageRank(const CommSpec& comm_spec,
                           const std::string& out_prefix, int fnum,
                           const ParallelEngineSpec& spec, double delta,
                           int mr) {
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
    using VertexMapType =
        GlobalVertexMap<OID_T, VID_T, SegmentedPartitioner<OID_T>>;
    using FRAG_T = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                            load_strategy, VertexMapType>;
    std::shared_ptr<FRAG_T> fragment =
        LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
    uint64_t local_ivnum = fragment->GetInnerVerticesNum();
    uint64_t local_ovnum = fragment->GetOuterVerticesNum();
    uint64_t total_ivnum, total_ovnum;
    MPI_Allreduce(&local_ivnum, &total_ivnum, 1, MPI_UINT64_T, MPI_SUM,
                  comm_spec.comm());
    MPI_Allreduce(&local_ovnum, &total_ovnum, 1, MPI_UINT64_T, MPI_SUM,
                  comm_spec.comm());

    double avg_degree = static_cast<double>(FLAGS_edge_num) /
                        static_cast<double>(FLAGS_vertex_num);
    double rate =
        static_cast<double>(total_ovnum) / static_cast<double>(total_ivnum);

    bool push = false;
    if (avg_degree > 90) {
      // too dense
      push = false;
    } else if (rate < 3) {
      // not to many outer vertices
      push = true;
    } else if (avg_degree / rate > 20) {
      push = true;
    } else {
      push = false;
    }

    if (!push) {
      using AppType = PageRankOpt<FRAG_T>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, double, int>(fragment, app, comm_spec, spec,
                                            out_prefix, delta, mr);
    } else {
      using AppType = PageRankPushOpt<FRAG_T>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, double, int>(fragment, app, comm_spec, spec,
                                            out_prefix, delta, mr);
    }
  } else {
    graph_spec.set_rebalance(false, 0);
    using FRAG_T =
        ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T, load_strategy>;
    std::shared_ptr<FRAG_T> fragment =
        LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);

    uint64_t local_ivnum = fragment->GetInnerVerticesNum();
    uint64_t local_ovnum = fragment->GetOuterVerticesNum();
    uint64_t total_ivnum, total_ovnum;
    MPI_Allreduce(&local_ivnum, &total_ivnum, 1, MPI_UINT64_T, MPI_SUM,
                  comm_spec.comm());
    MPI_Allreduce(&local_ovnum, &total_ovnum, 1, MPI_UINT64_T, MPI_SUM,
                  comm_spec.comm());

    if (static_cast<double>(total_ovnum) >
        static_cast<double>(total_ivnum) * 3.2) {
      using AppType = PageRank<FRAG_T>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, double, int>(fragment, app, comm_spec, spec,
                                            out_prefix, delta, mr);
    } else {
      using AppType = PageRankPushOpt<FRAG_T>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, double, int>(fragment, app, comm_spec, spec,
                                            out_prefix, delta, mr);
    }
  }
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
void RunOpt() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);
#ifdef GRANULA
  std::string job_id = FLAGS_jobid;
  granula::startMonitorProcess(getpid());
  granula::operation grapeJob("grape", "Id.Unique", "Job", "Id.Unique");
  granula::operation loadGraph("grape", "Id.Unique", "LoadGraph", "Id.Unique");
  if (comm_spec.worker_id() == kCoordinatorRank) {
    std::cout << grapeJob.getOperationInfo("StartTime", grapeJob.getEpoch())
              << std::endl;
    std::cout << loadGraph.getOperationInfo("StartTime", loadGraph.getEpoch())
              << std::endl;
  }

  granula::linkNode(job_id);
  granula::linkProcess(getpid(), job_id);
#endif

#ifdef GRANULA
  if (comm_spec.worker_id() == kCoordinatorRank) {
    std::cout << loadGraph.getOperationInfo("EndTime", loadGraph.getEpoch())
              << std::endl;
  }
#endif
  // FIXME: no barrier apps. more manager? or use a dynamic-cast.
  std::string out_prefix = FLAGS_out_prefix;
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
  int fnum = comm_spec.fnum();
  std::string name = FLAGS_application;
  if (name == "sssp") {
    FLAGS_segmented_partition = true;
    FLAGS_rebalance = false;
    CreateAndQuery<OID_T, VID_T, VDATA_T, double, LoadStrategy::kOnlyOut,
                   SSSPOpt, OID_T>(comm_spec, out_prefix, fnum, spec,
                                   FLAGS_sssp_source);
  } else if (name == "bfs") {
    if (FLAGS_directed) {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = false;
      CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kBothOutIn,
                     BFSOpt, OID_T>(comm_spec, out_prefix, fnum, spec,
                                    FLAGS_bfs_source);
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = false;
      CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                     BFSOpt, OID_T>(comm_spec, out_prefix, fnum, spec,
                                    FLAGS_bfs_source);
    }
  } else if (name == "pagerank") {
    if (FLAGS_directed) {
      FLAGS_segmented_partition = false;
      CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kBothOutIn,
                     PageRankDirected, double, int>(
          comm_spec, out_prefix, fnum, spec, FLAGS_pr_d, FLAGS_pr_mr);
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = true;
      FLAGS_rebalance_vertex_factor = 0;
      RunUndirectedPageRank<OID_T, VID_T, VDATA_T, EmptyType,
                            LoadStrategy::kOnlyOut>(
          comm_spec, out_prefix, fnum, spec, FLAGS_pr_d, FLAGS_pr_mr);
    }
  } else if (name == "cdlp") {
    if (FLAGS_directed) {
      FLAGS_directed = false;
      FLAGS_segmented_partition = false;
      CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                     CDLPOpt, int>(comm_spec, out_prefix, fnum, spec,
                                   FLAGS_cdlp_mr);
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = true;
      FLAGS_rebalance_vertex_factor = 0;
      double avg_deg = static_cast<double>(FLAGS_edge_num) /
                       static_cast<double>(FLAGS_vertex_num);
      if (avg_deg > 80) {
        CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                       CDLPOptUDDense, int>(comm_spec, out_prefix, fnum, spec,
                                            FLAGS_cdlp_mr);

      } else {
        CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                       CDLPOptUD, int>(comm_spec, out_prefix, fnum, spec,
                                       FLAGS_cdlp_mr);
      }
    }
  } else if (name == "wcc") {
    FLAGS_directed = false;
    FLAGS_segmented_partition = true;
    FLAGS_rebalance = false;
    CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                   WCCOpt>(comm_spec, out_prefix, fnum, spec);
  } else if (name == "lcc") {
    if (FLAGS_directed) {
      FLAGS_segmented_partition = false;
      if (FLAGS_edge_num >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType,
                       LoadStrategy::kBothOutIn, LCCDirected64>(
            comm_spec, out_prefix, fnum, spec);
      } else {
        CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType,
                       LoadStrategy::kBothOutIn, LCCDirected32>(
            comm_spec, out_prefix, fnum, spec);
      }
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = true;
      FLAGS_rebalance_vertex_factor = 0;
      if (FLAGS_edge_num >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) * 2) {
        CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                       LCC64>(comm_spec, out_prefix, fnum, spec);
      } else {
        CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                       LCC32>(comm_spec, out_prefix, fnum, spec);
      }
    }
  } else if (name == "lcc_beta") {
    CHECK(!FLAGS_directed);
    if (FLAGS_edge_num >
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) * 2) {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                     LCCBeta64>(comm_spec, out_prefix, fnum, spec);
    } else {
      CreateAndQuery<OID_T, VID_T, VDATA_T, EmptyType, LoadStrategy::kOnlyOut,
                     LCCBeta32>(comm_spec, out_prefix, fnum, spec);
    }
  } else {
    LOG(FATAL) << "No avaiable application named [" << name << "].";
  }
#ifdef GRANULA
  granula::operation offloadGraph("grape", "Id.Unique", "OffloadGraph",
                                  "Id.Unique");
#endif

#ifdef GRANULA
  if (comm_spec.worker_id() == kCoordinatorRank) {
    std::cout << offloadGraph.getOperationInfo("StartTime",
                                               offloadGraph.getEpoch())
              << std::endl;

    std::cout << grapeJob.getOperationInfo("EndTime", grapeJob.getEpoch())
              << std::endl;
  }

  granula::stopMonitorProcess(getpid());
#endif
}

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_APP_OPT_H_
