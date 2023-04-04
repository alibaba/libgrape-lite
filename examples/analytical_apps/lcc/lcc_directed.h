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

#ifndef EXAMPLES_ANALYTICAL_APPS_LCC_LCC_DIRECTED_H_
#define EXAMPLES_ANALYTICAL_APPS_LCC_LCC_DIRECTED_H_

#include <grape/grape.h>

#include <vector>

#include "lcc/lcc_directed_context.h"

namespace grape {

/**
 * @brief An implementation of LCC (Local CLustering Coefficient), the version
 * in LDBC, which only works on undirected graphs.
 *
 * This version of LCC inherits ParallelAppBase. Messages can be sent in
 * parallel to the evaluation. This strategy improve performance by overlapping
 * the communication time and the evaluation time.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class LCCDirected : public ParallelAppBase<FRAG_T, LCCDirectedContext<FRAG_T>>,
            public ParallelEngine {
 public:
  INSTALL_PARALLEL_WORKER(LCCDirected<FRAG_T>, LCCDirectedContext<FRAG_T>, FRAG_T);
  using vertex_t = typename fragment_t::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kBothOutIn;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    using vid_t = typename context_t::vid_t;
    auto inner_vertices = frag.InnerVertices();

    messages.InitChannels(thread_num());

#ifdef PROFILING
    ctx.postprocess_time -= GetCurrentTime();
#endif

    ForEach(inner_vertices, [&frag, &messages, &ctx](int tid, vertex_t v) {
      auto& nbr_vec = ctx.complete_neighbor[v];
      std::vector<vid_t> msg_vec;
      auto oes = frag.GetOutgoingAdjList(v);
      for (auto& e : oes) {
        auto u = e.get_neighbor();
        nbr_vec.push_back(u);
        msg_vec.push_back(frag.Vertex2Gid(u));
      }
      auto ies = frag.GetIncomingAdjList(v);
      for (auto& e : ies) {
        auto u = e.get_neighbor();
        nbr_vec.push_back(u);
        msg_vec.push_back(frag.Vertex2Gid(u));
      }
      messages.SendMsgThroughEdges<fragment_t, std::vector<vid_t>>(frag, v, msg_vec, tid);
    });

#ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
#endif
    // Just in case we are running on single process and no messages will
    // be send. ForceContinue() ensure the computation
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    using vid_t = typename context_t::vid_t;

    auto inner_vertices = frag.InnerVertices();
#ifdef PROFILING
      ctx.preprocess_time -= GetCurrentTime();
#endif
      messages.ParallelProcess<fragment_t, std::vector<vid_t>>(
          thread_num(), frag,
          [&frag, &ctx](int tid, vertex_t u, const std::vector<vid_t>& msg) {
            auto& nbr_vec = ctx.complete_neighbor[u];
            for (auto gid : msg) {
              vertex_t v;
              if (frag.Gid2Vertex(gid, v)) {
                nbr_vec.push_back(v);
              }
            }
          });

      std::vector<DenseVertexSet<typename FRAG_T::vertices_t>> vertexsets(
          thread_num());

      ForEach(
          inner_vertices,
          [&vertexsets, &frag](int tid) {
            auto& ns = vertexsets[tid];
            ns.Init(frag.Vertices());
          },
          [&vertexsets, &ctx](int tid, vertex_t v) {
            auto& v0_nbr_set = vertexsets[tid];
            auto& v0_nbr_vec = ctx.complete_neighbor[v];
            std::vector<vertex_t> deduped_v0_nbr_vec;
            for (auto u : v0_nbr_vec) {
              if (v0_nbr_set.Exist(u)) {
                continue;
              } else {
                deduped_v0_nbr_vec.push_back(u);
                v0_nbr_set.Insert(u);
              }
            }
            ctx.global_degree[v] = deduped_v0_nbr_vec.size();
            int count = 0;
            for (auto u : deduped_v0_nbr_vec) {
              auto& v1_nbr_vec = ctx.complete_neighbor[u];
              for (auto w : v1_nbr_vec) {
                if (u < w) {
                  if (v0_nbr_set.Exist(w)) {
                    ++count;
                  }
                }
              }
            }
            ctx.tricnt[v] = count;
            for (auto u : deduped_v0_nbr_vec) {
              v0_nbr_set.Erase(u);
            }
          },
          [](int tid) {});

#ifdef PROFILING
      ctx.exec_time += GetCurrentTime();
      ctx.postprocess_time -= GetCurrentTime();
#endif
  }
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_LCC_LCC_DIRECTED_H_
