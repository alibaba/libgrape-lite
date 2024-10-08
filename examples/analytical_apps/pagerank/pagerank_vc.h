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

#ifndef EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_H_
#define EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_H_

#include <grape/grape.h>

#include "pagerank/pagerank_vc_context.h"

namespace grape {

template <typename T>
struct NumericSum {
  static T init() { return 0; }

  static void aggregate(T& a, const T& b) { a += b; }
};

template <typename FRAG_T>
class PageRankVC
    : public GatherScatterAppBase<FRAG_T, PageRankVCContext<FRAG_T>>,
      public ParallelEngine,
      public Communicator {
  using vertex_t = Vertex<typename FRAG_T::oid_t>;

 public:
  INSTALL_GATHER_SCATTER_WORKER(PageRankVC<FRAG_T>, PageRankVCContext<FRAG_T>,
                                FRAG_T)

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    if (ctx.max_round <= 0) {
      return;
    }

    ctx.step = 0;
    ctx.graph_vnum = frag.GetTotalVerticesNum();

    {
#ifdef PROFILING
      ctx.t0 -= GetCurrentTime();
#endif
      typename fragment_t::template both_vertex_array_t<int> degree(
          frag.Vertices(), 0);
#ifdef TRACKING_MEMORY
      // allocate degree array for both src and dst vertices
      MemoryTracker::GetInstance().allocate(frag.Vertices().size() *
                                            sizeof(int));
#endif
      int bucket_num = frag.GetBucketNum();
      int concurrency = thread_num();
      if (bucket_num < (concurrency / 2)) {
        ForEach(
            frag.GetEdges().begin(), frag.GetEdges().end(),
            [&](int tid, const typename fragment_t::edge_t& e) {
              atomic_add(degree[vertex_t(e.src)], 1);
              atomic_add(degree[vertex_t(e.dst)], 1);
            },
            4096);
      } else {
        ForEach(0, bucket_num,
                [&degree, &frag, bucket_num](int tid, int bucket_id) {
                  for (int i = 0; i < bucket_num; ++i) {
                    for (auto& e : frag.GetEdgesOfBucket(i, bucket_id)) {
                      degree[vertex_t(e.dst)] += 1;
                    }
                  }
                  for (int i = 0; i < bucket_num; ++i) {
                    for (auto& e : frag.GetEdgesOfBucket(bucket_id, i)) {
                      degree[vertex_t(e.src)] += 1;
                    }
                  }
                });
      }
#ifdef PROFILING
      ctx.t0 += GetCurrentTime();
#endif

      messages.GatherMasterVertices<fragment_t, int, NumericSum<int>>(
          frag, degree, ctx.master_degree);
#ifdef TRACKING_MEMORY
      // deallocate degree array for both src and dst vertices
      MemoryTracker::GetInstance().deallocate(frag.Vertices().size() *
                                              sizeof(int));
#endif
    }

    double p = 1.0 / ctx.graph_vnum;
    int64_t dangling_vnum_local = 0;
#ifdef PROFILING
    ctx.t1 -= GetCurrentTime();
#endif
    std::vector<int64_t> dangling_vnum_local_vec(thread_num(), 0);
    ForEach(frag.MasterVertices(), [&](int tid, vertex_t v) {
      if (ctx.master_degree[v] == 0) {
        ++dangling_vnum_local_vec[tid];
        ctx.master_result[v] = p;
      } else {
        ctx.master_result[v] = p / ctx.master_degree[v];
      }
    });
    for (auto x : dangling_vnum_local_vec) {
      dangling_vnum_local += x;
    }
#ifdef PROFILING
    ctx.t1 += GetCurrentTime();
#endif

    Sum(dangling_vnum_local, ctx.total_dangling_vnum);
    ctx.dangling_sum = p * ctx.total_dangling_vnum;

    messages.ScatterMasterVertices<fragment_t, double>(frag, ctx.master_result,
                                                       ctx.curr_result);
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    if (ctx.step == 0) {
      messages.AllocateGatherBuffers<fragment_t, double>(frag);
    }
    ++ctx.step;

    double base = (1.0 - ctx.delta) / ctx.graph_vnum +
                  ctx.delta * ctx.dangling_sum / ctx.graph_vnum;
    ctx.dangling_sum = base * ctx.total_dangling_vnum;

    ForEach(frag.Vertices(),
            [&ctx](int tid, vertex_t v) { ctx.next_result[v] = 0; });

    int bucket_num = frag.GetBucketNum();
    int concurrency = thread_num();

#ifdef PROFILING
    ctx.t2 -= GetCurrentTime();
#endif
    if (bucket_num < (concurrency / 2)) {
      ForEach(
          frag.GetEdges().begin(), frag.GetEdges().end(),
          [&ctx](int tid, const typename fragment_t::edge_t& e) {
            atomic_add(ctx.next_result[vertex_t(e.dst)],
                       ctx.curr_result[vertex_t(e.src)]);
            atomic_add(ctx.next_result[vertex_t(e.src)],
                       ctx.curr_result[vertex_t(e.dst)]);
          },
          4096);
    } else {
      ForEach(0, bucket_num, [&ctx, &frag, bucket_num](int tid, int bucket_id) {
        for (int i = 0; i < bucket_num; ++i) {
          for (auto& e : frag.GetEdgesOfBucket(i, bucket_id)) {
            ctx.next_result[vertex_t(e.dst)] +=
                ctx.curr_result[vertex_t(e.src)];
          }
        }
        for (int i = 0; i < bucket_num; ++i) {
          for (auto& e : frag.GetEdgesOfBucket(bucket_id, i)) {
            ctx.next_result[vertex_t(e.src)] +=
                ctx.curr_result[vertex_t(e.dst)];
          }
        }
      });
    }

#ifdef PROFILING
    ctx.t2 += GetCurrentTime();
#endif

    messages.GatherMasterVertices<fragment_t, double, NumericSum<double>>(
        frag, ctx.next_result, ctx.master_result);

    if (ctx.step != ctx.max_round) {
#ifdef PROFILING
      ctx.t1 -= GetCurrentTime();
#endif
      ForEach(frag.MasterVertices(), [&ctx, base](int tid, vertex_t v) {
        if (ctx.master_degree[v] > 0) {
          ctx.master_result[v] =
              (base + ctx.delta * ctx.master_result[v]) / ctx.master_degree[v];
        } else {
          ctx.master_result[v] = base;
        }
      });
#ifdef PROFILING
      ctx.t1 += GetCurrentTime();
#endif

      messages.ScatterMasterVertices<fragment_t, double>(
          frag, ctx.master_result, ctx.curr_result);
      messages.ForceContinue();
    } else {
#ifdef PROFILING
      ctx.t1 -= GetCurrentTime();
#endif
      ForEach(frag.MasterVertices(), [&ctx, base](int tid, vertex_t v) {
        ctx.master_result[v] = ctx.master_result[v] * ctx.delta + base;
      });
#ifdef PROFILING
      ctx.t1 += GetCurrentTime();
#endif
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_H_
