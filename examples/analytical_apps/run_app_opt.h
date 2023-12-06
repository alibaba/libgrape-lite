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

template <LoadStrategy load_strategy>
void RunUndirectedPageRankOpt(const CommSpec& comm_spec,
                              const std::string& out_prefix,
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
        GlobalVertexMap<int64_t, uint32_t, SegmentedPartitioner<int64_t>>;
    using FRAG_T =
        ImmutableEdgecutFragment<int64_t, uint32_t, EmptyType, EmptyType,
                                 load_strategy, VertexMapType>;
    std::shared_ptr<FRAG_T> fragment =
        LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
    bool push;
    if (fragment->fnum() >= 8) {
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

      if (rate < 0.5) {
        // not to many outer vertices
        push = true;
      } else if (avg_degree > 60) {
        // dense
        push = true;
      } else {
        push = false;
      }
    } else {
      push = true;
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
    using FRAG_T = ImmutableEdgecutFragment<int64_t, uint32_t, EmptyType,
                                            EmptyType, load_strategy>;
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

template <typename VERTEX_MAP_T>
std::pair<int64_t, int64_t> get_min_max_id(const VERTEX_MAP_T& vm) {
  fid_t fnum = vm.GetFragmentNum();
  using vid_t = typename VERTEX_MAP_T::vid_t;
  int thread_num = std::thread::hardware_concurrency();
  std::vector<int64_t> min_ids(thread_num, std::numeric_limits<int64_t>::max());
  std::vector<int64_t> max_ids(thread_num, std::numeric_limits<int64_t>::min());
  for (fid_t i = 0; i != fnum; ++i) {
    vid_t ivnum = vm.GetInnerVertexSize(i);
    std::atomic<vid_t> cur(0);
    const vid_t chunk = 4096;
    std::vector<std::thread> threads;
    for (int j = 0; j < thread_num; ++j) {
      threads.emplace_back(
          [&](int tid) {
            vid_t begin = std::min(cur.fetch_add(chunk), ivnum);
            vid_t end = std::min(begin + chunk, ivnum);
            int64_t local_min = std::numeric_limits<int64_t>::max();
            int64_t local_max = std::numeric_limits<int64_t>::min();
            if (begin == end) {
              min_ids[tid] = std::min(min_ids[tid], local_min);
              max_ids[tid] = std::max(max_ids[tid], local_max);
              return;
            }
            while (begin != end) {
              int64_t oid;
              CHECK(vm.GetOid(i, begin++, oid));
              local_max = std::max(local_max, oid);
              local_min = std::min(local_min, oid);
            }
          },
          j);
    }
    for (auto& thrd : threads) {
      thrd.join();
    }
  }
  return std::make_pair(*std::min_element(min_ids.begin(), min_ids.end()),
                        *std::max_element(max_ids.begin(), max_ids.end()));
}

bool is_int32(int64_t v) {
  return v <= std::numeric_limits<int32_t>::max() &&
         v >= std::numeric_limits<int32_t>::min();
}

void RunDirectedCDLP(const CommSpec& comm_spec, const std::string& out_prefix,
                     const ParallelEngineSpec& spec) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }

  using FRAG_T = ImmutableEdgecutFragment<int64_t, uint32_t, EmptyType,
                                          EmptyType, LoadStrategy::kOnlyOut>;

  std::shared_ptr<FRAG_T> fragment =
      LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);

  std::pair<int64_t, int64_t> min_max_id =
      get_min_max_id(*fragment->GetVertexMap());
  if (is_int32(min_max_id.first) && is_int32(min_max_id.second)) {
    using AppType = CDLPOpt<FRAG_T, int32_t>;
    auto app = std::make_shared<AppType>();
    DoQuery<FRAG_T, AppType, int>(fragment, app, comm_spec, spec, out_prefix,
                                  FLAGS_cdlp_mr);
  } else {
    using AppType = CDLPOpt<FRAG_T, int64_t>;
    auto app = std::make_shared<AppType>();
    DoQuery<FRAG_T, AppType, int>(fragment, app, comm_spec, spec, out_prefix,
                                  FLAGS_cdlp_mr);
  }
}

void RunUndirectedCDLP(const CommSpec& comm_spec, const std::string& out_prefix,
                       const ParallelEngineSpec& spec) {
  timer_next("load graph");
  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();
  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    graph_spec.set_deserialize(true, FLAGS_serialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }

  using VertexMapType =
      GlobalVertexMap<int64_t, uint32_t, SegmentedPartitioner<int64_t>>;
  using FRAG_T =
      ImmutableEdgecutFragment<int64_t, uint32_t, EmptyType, EmptyType,
                               LoadStrategy::kOnlyOut, VertexMapType>;

  std::shared_ptr<FRAG_T> fragment =
      LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);

  double avg_degree = static_cast<double>(FLAGS_edge_num) /
                      static_cast<double>(FLAGS_vertex_num);
  std::pair<int64_t, int64_t> min_max_id =
      get_min_max_id(*fragment->GetVertexMap());
  if (is_int32(min_max_id.first) && is_int32(min_max_id.second)) {
    if (avg_degree > 256) {
      using AppType = CDLPOptUDDense<FRAG_T, int32_t>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, int>(fragment, app, comm_spec, spec, out_prefix,
                                    FLAGS_cdlp_mr);
    } else {
      using AppType = CDLPOptUD<FRAG_T, int32_t>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, int>(fragment, app, comm_spec, spec, out_prefix,
                                    FLAGS_cdlp_mr);
    }
  } else {
    if (avg_degree > 256) {
      using AppType = CDLPOptUDDense<FRAG_T, int64_t>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, int>(fragment, app, comm_spec, spec, out_prefix,
                                    FLAGS_cdlp_mr);
    } else {
      using AppType = CDLPOptUD<FRAG_T, int64_t>;
      auto app = std::make_shared<AppType>();
      DoQuery<FRAG_T, AppType, int>(fragment, app, comm_spec, spec, out_prefix,
                                    FLAGS_cdlp_mr);
    }
  }
}

template <typename EDATA_T, LoadStrategy load_strategy,
          template <class> class APP_T, typename... Args>
void CreateAndQueryOpt(const CommSpec& comm_spec, const std::string& out_prefix,
                       const ParallelEngineSpec& spec, Args... args) {
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
        GlobalVertexMap<int64_t, uint32_t, SegmentedPartitioner<int64_t>>;
    using FRAG_T =
        ImmutableEdgecutFragment<int64_t, uint32_t, grape::EmptyType, EDATA_T,
                                 load_strategy, VertexMapType>;
    std::shared_ptr<FRAG_T> fragment =
        LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
    using AppType = APP_T<FRAG_T>;
    auto app = std::make_shared<AppType>();
    DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                      out_prefix, args...);
  } else {
    graph_spec.set_rebalance(false, 0);
    using FRAG_T = ImmutableEdgecutFragment<int64_t, uint32_t, grape::EmptyType,
                                            EDATA_T, load_strategy>;
    std::shared_ptr<FRAG_T> fragment =
        LoadGraph<FRAG_T>(FLAGS_efile, FLAGS_vfile, comm_spec, graph_spec);
    using AppType = APP_T<FRAG_T>;
    auto app = std::make_shared<AppType>();
    DoQuery<FRAG_T, AppType, Args...>(fragment, app, comm_spec, spec,
                                      out_prefix, args...);
  }
}

void RunOpt() {
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == kCoordinatorRank;
  timer_start(is_coordinator);

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
  std::string name = FLAGS_application;
  if (name == "sssp") {
    FLAGS_segmented_partition = true;
    FLAGS_rebalance = false;
    CreateAndQueryOpt<double, LoadStrategy::kOnlyOut, SSSPOpt, int64_t>(
        comm_spec, out_prefix, spec, FLAGS_sssp_source);
  } else if (name == "bfs") {
    if (FLAGS_directed) {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = false;
      CreateAndQueryOpt<EmptyType, LoadStrategy::kBothOutIn, BFSOpt, int64_t>(
          comm_spec, out_prefix, spec, FLAGS_bfs_source);
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = false;
      CreateAndQueryOpt<EmptyType, LoadStrategy::kOnlyOut, BFSOpt, int64_t>(
          comm_spec, out_prefix, spec, FLAGS_bfs_source);
    }
  } else if (name == "pagerank") {
    if (FLAGS_directed) {
      FLAGS_segmented_partition = false;
      CreateAndQueryOpt<EmptyType, LoadStrategy::kBothOutIn, PageRankDirected,
                        double, int>(comm_spec, out_prefix, spec, FLAGS_pr_d,
                                     FLAGS_pr_mr);
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = true;
      FLAGS_rebalance_vertex_factor = 0;
      RunUndirectedPageRankOpt<LoadStrategy::kOnlyOut>(
          comm_spec, out_prefix, spec, FLAGS_pr_d, FLAGS_pr_mr);
    }
  } else if (name == "cdlp") {
    if (FLAGS_directed) {
      FLAGS_directed = false;
      FLAGS_segmented_partition = false;
      RunDirectedCDLP(comm_spec, out_prefix, spec);
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = true;
      FLAGS_rebalance_vertex_factor = 0;
      RunUndirectedCDLP(comm_spec, out_prefix, spec);
    }
  } else if (name == "wcc") {
    FLAGS_directed = false;
    FLAGS_segmented_partition = true;
    FLAGS_rebalance = false;
    CreateAndQueryOpt<EmptyType, LoadStrategy::kOnlyOut, WCCOpt>(
        comm_spec, out_prefix, spec);
  } else if (name == "lcc") {
    if (FLAGS_directed) {
      FLAGS_segmented_partition = false;
      if (FLAGS_edge_num >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        CreateAndQueryOpt<EmptyType, LoadStrategy::kBothOutIn, LCCDirected64>(
            comm_spec, out_prefix, spec);
      } else {
        CreateAndQueryOpt<EmptyType, LoadStrategy::kBothOutIn, LCCDirected32>(
            comm_spec, out_prefix, spec);
      }
    } else {
      FLAGS_segmented_partition = true;
      FLAGS_rebalance = true;
      FLAGS_rebalance_vertex_factor = 0;
      if (FLAGS_edge_num >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) * 2) {
        CreateAndQueryOpt<EmptyType, LoadStrategy::kOnlyOut, LCC64>(
            comm_spec, out_prefix, spec);
      } else {
        CreateAndQueryOpt<EmptyType, LoadStrategy::kOnlyOut, LCC32>(
            comm_spec, out_prefix, spec);
      }
    }
  } else if (name == "lcc_beta") {
    CHECK(!FLAGS_directed);
    if (FLAGS_edge_num >
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) * 2) {
      CreateAndQueryOpt<EmptyType, LoadStrategy::kOnlyOut, LCCBeta64>(
          comm_spec, out_prefix, spec);
    } else {
      CreateAndQueryOpt<EmptyType, LoadStrategy::kOnlyOut, LCCBeta32>(
          comm_spec, out_prefix, spec);
    }
  } else {
    LOG(FATAL) << "No avaiable application named [" << name << "].";
  }
}

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_APP_OPT_H_
