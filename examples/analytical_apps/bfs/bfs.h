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

#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_H_

#include <grape/grape.h>

#include "bfs/bfs_context.h"

namespace grape {

/**
 * @brief An implementation of BFS, the version in LDBC, which can work
 * on both directed or undirected graph.
 *
 * This version of BFS inherits ParallelAppBase. Messages can be sent in
 * parallel to the evaluation. This strategy improve performance by overlapping
 * the communication time and the evaluation time.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class BFS : public ParallelAppBase<FRAG_T, BFSContext<FRAG_T>>,
            public ParallelEngine {
 public:
  INSTALL_PARALLEL_WORKER(BFS<FRAG_T>, BFSContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  static constexpr bool need_split_edges = true;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    using depth_type = typename context_t::depth_type;

    messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);

    ctx.current_depth = 1;

    vertex_t source;
    bool native_source = frag.GetInnerVertex(ctx.source_id, source);

    auto inner_vertices = frag.InnerVertices();

    // init double buffer which contains updated vertices using bitmap
    ctx.curr_inner_updated.Init(inner_vertices, GetThreadPool());
    ctx.next_inner_updated.Init(inner_vertices, GetThreadPool());

#ifdef PROFILING
    ctx.exec_time -= GetCurrentTime();
#endif

    auto& channel_0 = messages.Channels()[0];

    // run first round BFS, update unreached vertices
    if (native_source) {
      ctx.partial_result[source] = 0;
      auto oes = frag.GetOutgoingAdjList(source);
      for (auto& e : oes) {
        auto u = e.get_neighbor();
        if (ctx.partial_result[u] == std::numeric_limits<depth_type>::max()) {
          ctx.partial_result[u] = 1;
          if (frag.IsOuterVertex(u)) {
            channel_0.SyncStateOnOuterVertex<fragment_t>(frag, u);
          } else {
            ctx.curr_inner_updated.Insert(u);
          }
        }
      }
    }

#ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
#endif

    messages.ForceContinue();

#ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
#endif
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    using depth_type = typename context_t::depth_type;

    auto& channels = messages.Channels();

    depth_type next_depth = ctx.current_depth + 1;
    int thrd_num = thread_num();
    ctx.next_inner_updated.ParallelClear(GetThreadPool());

#ifdef PROFILING
    ctx.preprocess_time -= GetCurrentTime();
#endif

    // process received messages and update depth
    messages.ParallelProcess<fragment_t, EmptyType>(
        thrd_num, frag, [&ctx](int tid, vertex_t v, EmptyType) {
          if (ctx.partial_result[v] == std::numeric_limits<depth_type>::max()) {
            ctx.partial_result[v] = ctx.current_depth;
            ctx.curr_inner_updated.Insert(v);
          }
        });

#ifdef PROFILING
    ctx.preprocess_time += GetCurrentTime();
    ctx.exec_time -= GetCurrentTime();
#endif

    // sync messages to other workers
    double rate = 0;
    if (ctx.avg_degree > 10) {
      auto ivnum = frag.GetInnerVerticesNum();
      rate = static_cast<double>(
                 ctx.curr_inner_updated.ParallelCount(GetThreadPool())) /
             static_cast<double>(ivnum);
      if (rate > 0.1) {
        auto inner_vertices = frag.InnerVertices();
        auto outer_vertices = frag.OuterVertices();
        ForEach(outer_vertices, [next_depth, &frag, &ctx, &channels](
                                    int tid, vertex_t v) {
          if (ctx.partial_result[v] == std::numeric_limits<depth_type>::max()) {
            auto ies = frag.GetIncomingAdjList(v);
            for (auto& e : ies) {
              auto u = e.get_neighbor();
              if (ctx.curr_inner_updated.Exist(u)) {
                ctx.partial_result[v] = next_depth;
                channels[tid].SyncStateOnOuterVertex<fragment_t>(frag, v);
                break;
              }
            }
          }
        });
        ForEach(inner_vertices, [next_depth, &frag, &ctx](int tid, vertex_t v) {
          if (ctx.partial_result[v] == std::numeric_limits<depth_type>::max()) {
            auto oes = frag.GetOutgoingInnerVertexAdjList(v);
            for (auto& e : oes) {
              auto u = e.get_neighbor();
              if (ctx.curr_inner_updated.Exist(u)) {
                ctx.partial_result[v] = next_depth;
                ctx.next_inner_updated.Insert(v);
                break;
              }
            }
          }
        });
      } else {
        ForEach(ctx.curr_inner_updated, [next_depth, &frag, &ctx, &channels](
                                            int tid, vertex_t v) {
          auto oes = frag.GetOutgoingAdjList(v);
          for (auto& e : oes) {
            auto u = e.get_neighbor();
            if (ctx.partial_result[u] ==
                std::numeric_limits<depth_type>::max()) {
              ctx.partial_result[u] = next_depth;
              if (frag.IsOuterVertex(u)) {
                channels[tid].SyncStateOnOuterVertex<fragment_t>(frag, u);
              } else {
                ctx.next_inner_updated.Insert(u);
              }
            }
          }
        });
      }
    } else {
      ForEach(ctx.curr_inner_updated, [next_depth, &frag, &ctx, &channels](
                                          int tid, vertex_t v) {
        auto oes = frag.GetOutgoingAdjList(v);
        for (auto& e : oes) {
          auto u = e.get_neighbor();
          if (ctx.partial_result[u] == std::numeric_limits<depth_type>::max()) {
            ctx.partial_result[u] = next_depth;
            if (frag.IsOuterVertex(u)) {
              channels[tid].SyncStateOnOuterVertex<fragment_t>(frag, u);
            } else {
              ctx.next_inner_updated.Insert(u);
            }
          }
        }
      });
    }

#ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
#endif

    ctx.current_depth = next_depth;
    if (!ctx.next_inner_updated.Empty()) {
      messages.ForceContinue();
    }

    ctx.next_inner_updated.Swap(ctx.curr_inner_updated);

#ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
#endif
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_H_
