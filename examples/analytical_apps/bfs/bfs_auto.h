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

#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_AUTO_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_AUTO_H_

#include <queue>

#include <grape/grape.h>

#include "bfs/bfs_auto_context.h"

namespace grape {
/**
 * @brief An implementation of BFS without using explicit message-passing APIs,
 * the version in LDBC, which can work on both directed or undirected graph.
 *
 * This is the auto-parallel version inherit AutoAppBase. In this version, users
 * plug sequential algorithms for PEval and IncEval, libgrape-lite parallelizes
 * them in the distributed setting. Users are not aware of messages.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class BFSAuto : public AutoAppBase<FRAG_T, BFSAutoContext<FRAG_T>> {
  INSTALL_AUTO_WORKER(BFSAuto<FRAG_T>, BFSAutoContext<FRAG_T>, FRAG_T)

 private:
  using vertex_t = typename fragment_t::vertex_t;
  void LocalBFS(const fragment_t& frag, context_t& ctx,
                std::queue<vertex_t>& que) {
    while (!que.empty()) {
      vertex_t u = que.front();
      ctx.partial_result.Reset(u);
      que.pop();

      auto new_depth = ctx.partial_result[u] + 1;
      auto oes = frag.GetOutgoingAdjList(u);

      // set depth for neighbors with current depth + 1
      for (auto& e : oes) {
        vertex_t v = e.get_neighbor();
        if (ctx.partial_result[v] > new_depth) {
          ctx.partial_result.SetValue(v, new_depth);
          if (frag.IsInnerVertex(v)) {
            que.push(v);
          }
        }
      }
    }
  }

 public:
  void PEval(const fragment_t& frag, context_t& ctx) {
    vertex_t source;
    bool local_source = frag.GetInnerVertex(ctx.source_id, source);
    if (!local_source) {
      return;
    }

    ctx.partial_result.SetValue(source, 0);

    // enqueue source vertex and run a round BFS
    std::queue<vertex_t> que;
    que.push(source);

    LocalBFS(frag, ctx, que);
  }

  void IncEval(const fragment_t& frag, context_t& ctx) {
    auto inner_vertices = frag.InnerVertices();

    // filter changed vertices and enqueue, then run BFS until converged
    std::queue<vertex_t> que;
    for (auto v : inner_vertices) {
      if (ctx.partial_result.IsUpdated(v)) {
        que.push(v);
      }
    }

    LocalBFS(frag, ctx, que);
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_AUTO_H_
