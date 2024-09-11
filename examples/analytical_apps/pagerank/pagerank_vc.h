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
class PageRankVC : public VCAppBase<FRAG_T, PageRankVCContext<FRAG_T>>,
                   public ParallelEngine,
                   public Communicator {
  using vertex_t = Vertex<typename FRAG_T::oid_t>;

 public:
  INSTALL_VC_WORKER(PageRankVC<FRAG_T>, PageRankVCContext<FRAG_T>, FRAG_T)

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    if (ctx.max_round <= 0) {
      return;
    }

    ctx.step = 0;
    ctx.graph_vnum = frag.GetTotalVerticesNum();

    {
      typename fragment_t::template both_vertex_array_t<int> degree(
          frag.GetVertices(), 0);
      for (auto& e : frag.GetEdges()) {
        ++degree[vertex_t(e.src)];
        ++degree[vertex_t(e.dst)];
      }

      messages.GatherMasterVertices<fragment_t, int, NumericSum<int>>(
          frag, degree, ctx.master_degree);
    }

    int64_t dangling_vnum_local = 0;
    double p = 1.0 / ctx.graph_vnum;
    for (auto v : frag.GetMasterVertices()) {
      if (ctx.master_degree[v] == 0) {
        ++dangling_vnum_local;
        ctx.master_result[v] = p;
      } else {
        ctx.master_result[v] = p / ctx.master_degree[v];
      }
    }

    Sum(dangling_vnum_local, ctx.total_dangling_vnum);
    ctx.dangling_sum = p * ctx.total_dangling_vnum;

    messages.ScatterMasterVertices<fragment_t, double>(frag, ctx.master_result,
                                                       ctx.curr_result);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    ++ctx.step;

    double base = (1.0 - ctx.delta) / ctx.graph_vnum +
                  ctx.delta * ctx.dangling_sum / ctx.graph_vnum;
    ctx.dangling_sum = base * ctx.total_dangling_vnum;

    if (ctx.step != ctx.max_round) {
      for (auto& e : frag.GetEdges()) {
        ctx.next_result[vertex_t(e.dst)] += ctx.curr_result[vertex_t(e.src)];
        ctx.next_result[vertex_t(e.src)] += ctx.curr_result[vertex_t(e.dst)];
      }

      messages.GatherMasterVertices<fragment_t, double, NumericSum<double>>(
          frag, ctx.next_result, ctx.master_result);
      for (auto v : frag.GetMasterVertices()) {
        if (ctx.master_degree[v] > 0) {
          ctx.master_result[v] =
              (base + ctx.delta * ctx.master_result[v]) / ctx.master_degree[v];
        } else {
          ctx.master_result[v] = base;
        }
      }

      messages.ScatterMasterVertices<fragment_t, double>(
          frag, ctx.master_result, ctx.curr_result);
    } else {
      for (auto& e : frag.GetEdges()) {
        ctx.next_result[vertex_t(e.dst)] += ctx.curr_result[vertex_t(e.src)];
        ctx.next_result[vertex_t(e.src)] += ctx.curr_result[vertex_t(e.dst)];
      }

      messages.GatherMasterVertices<fragment_t, double, NumericSum<double>>(
          frag, ctx.next_result, ctx.master_result);
      for (auto v : frag.GetMasterVertices()) {
        ctx.master_result[v] = ctx.master_result[v] * ctx.delta + base;
      }

      messages.ForceTerminate();
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_H_
