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

#ifndef EXAMPLES_ANALYTICAL_APPS_LCC_LCC_H_
#define EXAMPLES_ANALYTICAL_APPS_LCC_LCC_H_

#include <grape/grape.h>

#include <vector>

#include "lcc/lcc_context.h"
#include "grape/utils/varint.h"

namespace grape {

template <typename T>
class LCCRefVector {
 public:
  LCCRefVector() : p_(nullptr), limit_(nullptr) {}
  ~LCCRefVector() {}

  void reset(const T* p, size_t size) {
    p_ = p;
    limit_ = p + size;
  }

  bool pop(T& val) {
    if (p_ == limit_) {
      return false;
    }
    val = *p_++;
    return true;
  }

 private:
  const T* p_;
  const T* limit_;
};

template <typename T>
OutArchive& operator>>(OutArchive& arc, LCCRefVector<T>& vec) {
  size_t size;
  arc >> size;
  vec.reset(static_cast<const T*>(arc.GetBytes(size * sizeof(T))), size);
  return arc;
}

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
template <typename FRAG_T, typename COUNT_T=uint32_t>
class LCC : public ParallelAppBase<FRAG_T, LCCContext<FRAG_T, COUNT_T>>,
            public ParallelEngine {
#if 0
  using VecOutType = std::vector<typename FRAG_T::vid_t>;
  using VecInType = LCCRefVector<typename FRAG_T::vid_t>;
#else
  using VecOutType = DeltaVarintEncoder<typename FRAG_T::vid_t>;
  using VecInType = DeltaVarintDecoder<typename FRAG_T::vid_t>;
#endif
 public:
  using fragment_t = FRAG_T;
  using context_t = LCCContext<FRAG_T, COUNT_T>;
  using message_manager_t = ParallelMessageManager;
  using worker_t = ParallelWorker<LCC<FRAG_T, COUNT_T>>;

  static constexpr bool sort_neighbor_by_global_id = true;

  virtual ~LCC() {}

  static std::shared_ptr<worker_t> CreateWorker(std::shared_ptr<LCC<FRAG_T, COUNT_T>> app, std::shared_ptr<FRAG_T> frag) {
    return std::shared_ptr<worker_t>(new worker_t(app, frag));
  }

  using vertex_t = typename fragment_t::vertex_t;
  using count_t = COUNT_T;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();

    messages.InitChannels(thread_num());

    ctx.stage = 0;

#ifdef PROFILING
    ctx.postprocess_time -= GetCurrentTime();
#endif

    // Each vertex scatter its own out degree.
    ForEach(inner_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
      ctx.global_degree[v] = frag.GetLocalOutDegree(v);
      messages.SendMsgThroughOEdges<fragment_t, int>(frag, v,
                                                     ctx.global_degree[v], tid);
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
    auto outer_vertices = frag.OuterVertices();

    if (ctx.stage == 0) {
      ctx.stage = 1;
#ifdef PROFILING
      ctx.preprocess_time -= GetCurrentTime();
#endif
      messages.ParallelProcess<fragment_t, int>(
          thread_num(), frag,
          [&ctx](int tid, vertex_t u, int msg) { ctx.global_degree[u] = msg; });

#ifdef PROFILING
      ctx.preprocess_time += GetCurrentTime();
      ctx.exec_time -= GetCurrentTime();
#endif
      if (ctx.degree_threshold == std::numeric_limits<int>::max()) {
        ForEach(inner_vertices,
                [&frag, &ctx, &messages](int tid, vertex_t v) {
                  vid_t u_gid, v_gid;
                  auto& nbr_vec = ctx.complete_neighbor[v];
                  int degree = ctx.global_degree[v];
                  nbr_vec.reserve(degree);
                  auto es = frag.GetOutgoingAdjList(v);
                  VecOutType msg_vec;
                  msg_vec.reserve(degree);
                  for (auto& e : es) {
                    auto u = e.get_neighbor();
                    if (ctx.global_degree[u] < ctx.global_degree[v]) {
                      nbr_vec.push_back(u);
                      msg_vec.push_back(frag.Vertex2Gid(u));
                    } else if (ctx.global_degree[u] == ctx.global_degree[v]) {
                      u_gid = frag.Vertex2Gid(u);
                      v_gid = frag.GetInnerVertexGid(v);
                      if (v_gid > u_gid) {
                        nbr_vec.push_back(u);
                        msg_vec.push_back(u_gid);
                      }
                    }
                  }
                  messages.SendMsgThroughOEdges<fragment_t, std::vector<vid_t>>(
                      frag, v, msg_vec, tid);
                });
      } else {
        ForEach(inner_vertices,
                [this, &frag, &ctx, &messages](int tid, vertex_t v) {
                  if (filterByDegree(frag, ctx, v)) {
                    return;
                  }
                  vid_t u_gid, v_gid;
                  auto& nbr_vec = ctx.complete_neighbor[v];
                  int degree = ctx.global_degree[v];
                  nbr_vec.reserve(degree);
                  auto es = frag.GetOutgoingAdjList(v);
                  VecOutType msg_vec;
                  msg_vec.reserve(degree);
                  for (auto& e : es) {
                    auto u = e.get_neighbor();
                    if (ctx.global_degree[u] < ctx.global_degree[v]) {
                      nbr_vec.push_back(u);
                      msg_vec.push_back(frag.Vertex2Gid(u));
                    } else if (ctx.global_degree[u] == ctx.global_degree[v]) {
                      u_gid = frag.Vertex2Gid(u);
                      v_gid = frag.GetInnerVertexGid(v);
                      if (v_gid > u_gid) {
                        nbr_vec.push_back(u);
                        msg_vec.push_back(u_gid);
                      }
                    }
                  }
                  messages.SendMsgThroughOEdges<fragment_t, std::vector<vid_t>>(
                      frag, v, msg_vec, tid);
                });
      }

#ifdef PROFILING
      ctx.exec_time += GetCurrentTime();
      ctx.postprocess_time -= GetCurrentTime();
      ctx.postprocess_time += GetCurrentTime();
#endif
      messages.ForceContinue();
    } else if (ctx.stage == 1) {
      ctx.stage = 2;
#ifdef PROFILING
      ctx.preprocess_time -= GetCurrentTime();
#endif
      messages.ParallelProcess<fragment_t, VecInType>(
          thread_num(), frag,
          [&frag, &ctx](int tid, vertex_t u, VecInType& msg) {
            auto& nbr_vec = ctx.complete_neighbor[u];
            vid_t gid;
            while (msg.pop(gid)) {
              vertex_t v;
              if (frag.Gid2Vertex(gid, v)) {
                nbr_vec.push_back(v);
              }
            }
          });

#ifdef PROFILING
      ctx.preprocess_time += GetCurrentTime();
      ctx.exec_time -= GetCurrentTime();
#endif

      std::vector<DenseVertexSet<typename FRAG_T::vertices_t>> vertexsets(
          thread_num());

      if (ctx.degree_threshold == std::numeric_limits<int>::max()) {
        ForEach(
            inner_vertices,
            [&vertexsets, &frag](int tid) {
              auto& ns = vertexsets[tid];
              ns.Init(frag.Vertices());
            },
            [&vertexsets, &ctx](int tid, vertex_t v) {
              auto& v0_nbr_set = vertexsets[tid];
              auto& v0_nbr_vec = ctx.complete_neighbor[v];
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
                    ++v_count;
                    atomic_add(ctx.tricnt[w], static_cast<count_t>(1));
                  }
                }
                atomic_add(ctx.tricnt[u], u_count);
              }
              atomic_add(ctx.tricnt[v], v_count);
              for (auto u : v0_nbr_vec) {
                v0_nbr_set.Erase(u);
              }
            },
            [](int tid) {});
      } else {
        ForEach(
            inner_vertices,
            [&vertexsets, &frag](int tid) {
              auto& ns = vertexsets[tid];
              ns.Init(frag.Vertices());
            },
            [this, &vertexsets, &frag, &ctx](int tid, vertex_t v) {
              if (filterByDegree(frag, ctx, v)) {
                return;
              }
              auto& v0_nbr_set = vertexsets[tid];
              auto& v0_nbr_vec = ctx.complete_neighbor[v];
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
                    ++v_count;
                    atomic_add(ctx.tricnt[w], static_cast<count_t>(1));
                  }
                }
                atomic_add(ctx.tricnt[u], u_count);
              }
              atomic_add(ctx.tricnt[v], v_count);
              for (auto u : v0_nbr_vec) {
                v0_nbr_set.Erase(u);
              }
            },
            [](int tid) {});
      }

#ifdef PROFILING
      ctx.exec_time += GetCurrentTime();
      ctx.postprocess_time -= GetCurrentTime();
#endif

      ForEach(outer_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
        if (ctx.tricnt[v] != 0) {
          messages.SyncStateOnOuterVertex<fragment_t, count_t>(frag, v,
                                                               ctx.tricnt[v], tid);
        }
      });

#ifdef PROFILING
      ctx.postprocess_time += GetCurrentTime();
#endif
      messages.ForceContinue();
    } else if (ctx.stage == 2) {
      ctx.stage = 3;
#ifdef PROFILING
      ctx.preprocess_time -= GetCurrentTime();
#endif
      messages.ParallelProcess<fragment_t, count_t>(
          thread_num(), frag, [&ctx](int tid, vertex_t u, count_t deg) {
            atomic_add(ctx.tricnt[u], deg);
          });
#ifdef PROFILING
      ctx.preprocess_time += GetCurrentTime();
#endif

      // output result to context data
      auto& global_degree = ctx.global_degree;
      auto& tricnt = ctx.tricnt;
      auto& ctx_data = ctx.data();

      for (auto v : inner_vertices) {
        if (global_degree[v] == 0 || global_degree[v] == 1) {
          ctx_data[v] = 0;
        } else {
          double re = 2.0 * (static_cast<int64_t>(tricnt[v])) /
                      (static_cast<int64_t>(global_degree[v]) *
                       (static_cast<int64_t>(global_degree[v]) - 1));
          ctx_data[v] = re;
        }
      }
    }
  }
  bool filterByDegree(const fragment_t& frag, context_t& ctx, vertex_t v) {
    int degree = frag.GetLocalOutDegree(v);
    if (frag.directed()) {
      degree += frag.GetLocalInDegree(v);
    }
    if (degree > ctx.degree_threshold) {
      return true;
    }
    return false;
  }
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_LCC_LCC_H_
