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

#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_OPT_H_

#include <grape/grape.h>

#include "bfs/bfs_opt_context.h"

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
class BFSOpt : public ParallelAppBase<FRAG_T, BFSOptContext<FRAG_T>,
                                      ParallelMessageManagerOpt>,
               public ParallelEngine {
  INSTALL_PARALLEL_OPT_WORKER(BFSOpt<FRAG_T>, BFSOptContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  static constexpr bool need_split_edges = true;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    using depth_type = typename context_t::depth_type;

    messages.InitChannels(thread_num(), 32768, 32768);

    ctx.current_depth = 1;

    vertex_t source;
    bool native_source = frag.GetInnerVertex(ctx.source_id, source);

    auto inner_vertices = frag.InnerVertices();

    // init double buffer which contains updated vertices using bitmap
    ctx.curr_inner_updated.Init(inner_vertices, GetThreadPool());
    ctx.next_inner_updated.Init(inner_vertices, GetThreadPool());

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
            channel_0.template SyncStateOnOuterVertex<fragment_t>(frag, u);
          } else {
            ctx.curr_inner_updated.Insert(u);
          }
        }
      }
    }

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    using depth_type = typename context_t::depth_type;

    auto& channels = messages.Channels();

    depth_type next_depth = ctx.current_depth + 1;
    ctx.next_inner_updated.ParallelClear(GetThreadPool());

    // process received messages and update depth
    messages.ParallelProcess<fragment_t, EmptyType>(
        thread_num(), frag, [&ctx](int tid, vertex_t v, EmptyType) {
          if (ctx.partial_result[v] == std::numeric_limits<depth_type>::max()) {
            ctx.partial_result[v] = ctx.current_depth;
            ctx.curr_inner_updated.Insert(v);
          }
        });

    // sync messages to other workers
    auto ivnum = frag.GetInnerVerticesNum();
    auto active = ctx.curr_inner_updated.ParallelCount(GetThreadPool());
    double rate = static_cast<double>(active) / static_cast<double>(ivnum);
    if (rate > 0.005) {
      auto inner_vertices = frag.InnerVertices();
      auto outer_vertices = frag.OuterVertices();
      ForEach(outer_vertices, [next_depth, &frag, &ctx, &channels](int tid,
                                                                   vertex_t v) {
        if (ctx.partial_result[v] == std::numeric_limits<depth_type>::max()) {
          auto ies = frag.GetIncomingAdjList(v);
          for (auto& e : ies) {
            auto u = e.get_neighbor();
            if (ctx.curr_inner_updated.Exist(u)) {
              ctx.partial_result[v] = next_depth;
              channels[tid].template SyncStateOnOuterVertex<fragment_t>(frag,
                                                                        v);
              break;
            }
          }
        }
      });
      if (frag.directed()) {
        ForEach(inner_vertices, [next_depth, &frag, &ctx](int tid, vertex_t v) {
          if (ctx.partial_result[v] == std::numeric_limits<depth_type>::max()) {
            auto ies = frag.GetIncomingInnerVertexAdjList(v);
            for (auto& e : ies) {
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
      }
    } else if (active > 0) {
      ForEach(ctx.curr_inner_updated, [next_depth, &frag, &ctx, &channels](
                                          int tid, vertex_t v) {
        auto oes = frag.GetOutgoingAdjList(v);
        for (auto& e : oes) {
          auto u = e.get_neighbor();
          if (ctx.partial_result[u] == std::numeric_limits<depth_type>::max()) {
            ctx.partial_result[u] = next_depth;
            if (frag.IsOuterVertex(u)) {
              channels[tid].template SyncStateOnOuterVertex<fragment_t>(frag,
                                                                        u);
            } else {
              ctx.next_inner_updated.Insert(u);
            }
          }
        }
      });
    }

    ctx.current_depth = next_depth;
    if (!ctx.next_inner_updated.Empty()) {
      messages.ForceContinue();
    }

    ctx.next_inner_updated.Swap(ctx.curr_inner_updated);
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_OPT_H_
