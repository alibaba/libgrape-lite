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

#ifndef EXAMPLES_ANALYTICAL_APPS_KCORE_KCORE_H_
#define EXAMPLES_ANALYTICAL_APPS_KCORE_KCORE_H_

#include <grape/grape.h>

#include "kcore/kcore_context.h"

namespace grape {

/**
 * @brief An implementation of k-core, which can work on undirected graph.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class KCore : public ParallelAppBase<FRAG_T, KCoreContext<FRAG_T>,
                                     ParallelMessageManagerOpt>,
              public ParallelEngine {
  INSTALL_PARALLEL_OPT_WORKER(KCore<FRAG_T>, KCoreContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();

    ctx.curr_inner_updated.Init(frag.InnerVertices(), GetThreadPool());
    ctx.next_inner_updated.Init(frag.InnerVertices(), GetThreadPool());
    ctx.outer_updated.Init(frag.OuterVertices(), GetThreadPool());
    ctx.removed.Init(frag.InnerVertices(), GetThreadPool());

    messages.InitChannels(thread_num());

    ctx.reduced_degrees.Init(frag.Vertices(), 0);

    ForEach(inner_vertices, [&frag, &ctx](int tid, vertex_t v) {
      ctx.partial_result[v] = frag.GetLocalOutDegree(v);
      if (ctx.partial_result[v] < ctx.k) {
        ctx.removed.Insert(v);
        for (auto& e : frag.GetOutgoingAdjList(v)) {
          auto u = e.get_neighbor();
          atomic_add(ctx.reduced_degrees[u], 1);
          if (frag.IsOuterVertex(u)) {
            ctx.outer_updated.Insert(u);
          } else {
            ctx.curr_inner_updated.Insert(u);
          }
        }
      }
    });

    auto& channels = messages.Channels();
    ForEach(ctx.outer_updated, [&frag, &ctx, &channels](int tid, vertex_t v) {
      channels[tid].SyncStateOnOuterVertex<fragment_t, int>(
          frag, v, ctx.reduced_degrees[v]);
      ctx.reduced_degrees[v] = 0;
    });
    ctx.outer_updated.ParallelClear(GetThreadPool());

    if (!ctx.curr_inner_updated.Empty()) {
      messages.ForceContinue();
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto& channels = messages.Channels();

    messages.ParallelProcess<fragment_t, int>(
        thread_num(), frag, [&ctx](int tid, vertex_t v, int msg) {
          atomic_add(ctx.reduced_degrees[v], msg);
          ctx.curr_inner_updated.Insert(v);
        });

    ctx.next_inner_updated.ParallelClear(GetThreadPool());
    ForEach(ctx.curr_inner_updated,
            [&frag, &ctx, &channels](int tid, vertex_t v) {
              if (ctx.partial_result[v] >= ctx.k) {
                ctx.partial_result[v] -= ctx.reduced_degrees[v];
                ctx.reduced_degrees[v] = 0;

                if (ctx.partial_result[v] < ctx.k) {
                  ctx.next_inner_updated.Insert(v);
                }
              } else {
                ctx.partial_result[v] -= ctx.reduced_degrees[v];
                ctx.reduced_degrees[v] = 0;
              }
            });

    ctx.curr_inner_updated.ParallelClear(GetThreadPool());
    ForEach(ctx.next_inner_updated, [&frag, &ctx](int tid, vertex_t v) {
      for (auto& e : frag.GetOutgoingAdjList(v)) {
        auto u = e.get_neighbor();
        atomic_add(ctx.reduced_degrees[u], 1);
        if (frag.IsOuterVertex(u)) {
          ctx.outer_updated.Insert(u);
        } else {
          ctx.curr_inner_updated.Insert(u);
        }
      }
    });

    ForEach(ctx.outer_updated, [&frag, &ctx, &channels](int tid, vertex_t v) {
      channels[tid].SyncStateOnOuterVertex<fragment_t, int>(
          frag, v, ctx.reduced_degrees[v]);
      ctx.reduced_degrees[v] = 0;
    });
    ctx.outer_updated.ParallelClear(GetThreadPool());

    if (!ctx.curr_inner_updated.Empty()) {
      messages.ForceContinue();
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_KCORE_KCORE_H_