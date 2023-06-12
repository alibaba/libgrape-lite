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

#ifndef EXAMPLES_ANALYTICAL_APPS_LCC_LCC_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_LCC_LCC_OPT_H_

#include <grape/grape.h>

#include <vector>

#include "grape/utils/varint.h"
#include "lcc/lcc_opt_context.h"

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
template <typename FRAG_T, typename COUNT_T = uint32_t>
class LCCOpt : public ParallelAppBase<FRAG_T, LCCOptContext<FRAG_T, COUNT_T>,
                                      ParallelMessageManagerOpt>,
               public ParallelEngine {
  using VecOutType = DeltaVarintEncoder<typename FRAG_T::vid_t>;
  using VecInType = DeltaVarintDecoder<typename FRAG_T::vid_t>;

 public:
  using fragment_t = FRAG_T;
  using context_t = LCCOptContext<FRAG_T, COUNT_T>;
  using message_manager_t = ParallelMessageManagerOpt;
  using worker_t = ParallelWorkerOpt<LCCOpt<FRAG_T, COUNT_T>>;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using count_t = COUNT_T;
  using tricnt_list_t = typename fragment_t::template vertex_array_t<count_t>;

  virtual ~LCCOpt() {}

  static std::shared_ptr<worker_t> CreateWorker(
      std::shared_ptr<LCCOpt<FRAG_T, COUNT_T>> app,
      std::shared_ptr<FRAG_T> frag) {
    return std::shared_ptr<worker_t>(new worker_t(app, frag));
  }

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();

    messages.InitChannels(thread_num());

    ctx.stage = 0;

    // Each vertex scatter its own out degree.
    ForEach(inner_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
      ctx.global_degree[v] = frag.GetLocalOutDegree(v);
      messages.SendMsgThroughOEdges<fragment_t, int>(frag, v,
                                                     ctx.global_degree[v], tid);
    });

    // Just in case we are running on single process and no messages will
    // be send. ForceContinue() ensure the computation
    messages.ForceContinue();
  }

  count_t intersect_with_bs(const lcc_opt_impl::ref_vector<vertex_t>& small,
                            const lcc_opt_impl::ref_vector<vertex_t>& large,
                            tricnt_list_t& result) {
    count_t ret = 0;
    auto from = large.begin();
    auto to = large.end();
    for (auto v : small) {
      from = std::lower_bound(from, to, v);
      if (from == to) {
        return ret;
      }
      if (*from == v) {
        ++ret;
        ++from;
        atomic_add(result[v], static_cast<count_t>(1));
      }
    }
    return ret;
  }

  count_t intersect(const lcc_opt_impl::ref_vector<vertex_t>& lhs,
                    const lcc_opt_impl::ref_vector<vertex_t>& rhs,
                    tricnt_list_t& result) {
    if (lhs.empty() || rhs.empty()) {
      return 0;
    }
    vid_t v_size = lhs.size();
    vid_t u_size = rhs.size();
    if (static_cast<double>(v_size + u_size) <
        std::min<double>(v_size, u_size) *
            ilogb(std::max<double>(v_size, u_size))) {
      count_t count = 0;
      vid_t i = 0, j = 0;
      while (i < v_size && j < u_size) {
        if (lhs[i] == rhs[j]) {
          atomic_add(result[lhs[i]], static_cast<count_t>(1));
          ++count;
          ++i;
          ++j;
        } else if (lhs[i] < rhs[j]) {
          ++i;
        } else {
          ++j;
        }
      }
      return count;
    } else {
      if (v_size > u_size) {
        return intersect_with_bs(rhs, lhs, result);
      } else {
        return intersect_with_bs(lhs, rhs, result);
      }
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    using vid_t = typename context_t::vid_t;

    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    if (ctx.stage == 0) {
      ctx.stage = 1;
      messages.ParallelProcess<fragment_t, int>(
          thread_num(), frag,
          [&ctx](int tid, vertex_t u, int msg) { ctx.global_degree[u] = msg; });
      std::vector<size_t> max_degrees(thread_num(), 0);
      ctx.memory_pools.resize(thread_num());
      ForEach(inner_vertices, [&frag, &ctx, &messages, &max_degrees](
                                  int tid, vertex_t v) {
        vid_t v_gid_hash = IdHasher<vid_t>::hash(frag.GetInnerVertexGid(v));
        auto& pool = ctx.memory_pools[tid];
        auto& nbr_vec = ctx.complete_neighbor[v];
        int degree = ctx.global_degree[v];
        auto es = frag.GetOutgoingAdjList(v);
        static thread_local VecOutType msg_vec;
        msg_vec.clear();
        pool.reserve(es.Size());
        for (auto& e : es) {
          auto u = e.get_neighbor();
          if (ctx.global_degree[u] > degree) {
            pool.push_back(u);
            msg_vec.push_back(frag.Vertex2Gid(u));
          } else if (ctx.global_degree[u] == degree) {
            vid_t u_gid = frag.Vertex2Gid(u);
            if (v_gid_hash > IdHasher<vid_t>::hash(u_gid)) {
              pool.push_back(u);
              msg_vec.push_back(u_gid);
            }
          }
        }
        nbr_vec = pool.finish();
        if (nbr_vec.empty()) {
          return;
        }
        std::sort(nbr_vec.begin(), nbr_vec.end());
        if (nbr_vec.size() > max_degrees[tid]) {
          max_degrees[tid] = nbr_vec.size();
        }
        messages.SendMsgThroughOEdges<fragment_t, VecOutType>(frag, v, msg_vec,
                                                              tid);
      });
      size_t max_degree = 0;
      for (auto x : max_degrees) {
        max_degree = std::max(x, max_degree);
      }
      ctx.degree_x = max_degree * 4 / 10;
      messages.ForceContinue();
    } else if (ctx.stage == 1) {
      ctx.stage = 2;
      messages.ParallelProcess<fragment_t, VecInType>(
          thread_num(), frag,
          [&frag, &ctx](int tid, vertex_t u, VecInType& msg) {
            auto& pool = ctx.memory_pools[tid];
            auto& nbr_vec = ctx.complete_neighbor[u];
            vid_t gid;
            pool.reserve(ctx.global_degree[u]);
            while (msg.pop(gid)) {
              vertex_t v;
              if (frag.Gid2Vertex(gid, v)) {
                pool.push_back(v);
              }
            }
            nbr_vec = pool.finish();
            std::sort(nbr_vec.begin(), nbr_vec.end());
          });
      std::vector<DenseVertexSet<typename FRAG_T::vertices_t>> vertexsets(
          thread_num());
      for (auto& vs : vertexsets) {
        vs.Init(frag.Vertices());
      }
      ForEach(inner_vertices, [this, &ctx, &vertexsets](int tid, vertex_t v) {
        auto& v0_nbr_vec = ctx.complete_neighbor[v];
        if (v0_nbr_vec.size() <= 1) {
          return;
        } else if (v0_nbr_vec.size() <= ctx.degree_x) {
          count_t v_count = 0;
          for (auto u : v0_nbr_vec) {
            auto& v1_nbr_vec = ctx.complete_neighbor[u];
            count_t u_count = intersect(v0_nbr_vec, v1_nbr_vec, ctx.tricnt);
            atomic_add(ctx.tricnt[u], u_count);
            v_count += u_count;
          }
          atomic_add(ctx.tricnt[v], v_count);
        } else {
          auto& v0_nbr_set = vertexsets[tid];
          for (auto u : v0_nbr_vec) {
            v0_nbr_set.Insert(u);
          }
          count_t v_count = 0;
          for (auto u : v0_nbr_vec) {
            count_t u_count = 0;
            auto& v1_nbr_vec = ctx.complete_neighbor[u];
            for (auto w : v1_nbr_vec) {
              if (v0_nbr_set.Exist(w)) {
                ++u_count;
                atomic_add(ctx.tricnt[w], static_cast<count_t>(1));
              }
            }
            v_count += u_count;
            atomic_add(ctx.tricnt[u], u_count);
          }
          atomic_add(ctx.tricnt[v], v_count);
          for (auto u : v0_nbr_vec) {
            v0_nbr_set.Erase(u);
          }
        }
      });
      ForEach(outer_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
        if (ctx.tricnt[v] != 0) {
          messages.SyncStateOnOuterVertex<fragment_t, count_t>(
              frag, v, ctx.tricnt[v], tid);
        }
      });
      messages.ForceContinue();
    } else if (ctx.stage == 2) {
      ctx.stage = 3;
      messages.ParallelProcess<fragment_t, count_t>(
          thread_num(), frag, [&ctx](int tid, vertex_t u, count_t deg) {
            atomic_add(ctx.tricnt[u], deg);
          });

      // output result to context data
      auto& global_degree = ctx.global_degree;
      auto& tricnt = ctx.tricnt;
      auto& ctx_data = ctx.data();

      ForEach(inner_vertices, [&](int tid, vertex_t v) {
        if (global_degree[v] == 0 || global_degree[v] == 1) {
          ctx_data[v] = 0;
        } else {
          double re = 2.0 * (static_cast<int64_t>(tricnt[v])) /
                      (static_cast<int64_t>(global_degree[v]) *
                       (static_cast<int64_t>(global_degree[v]) - 1));
          ctx_data[v] = re;
        }
      });
    }
  }
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_LCC_LCC_OPT_H_
