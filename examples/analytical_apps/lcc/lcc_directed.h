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

#include "lcc/lcc.h"
#include "lcc/lcc_directed_context.h"

namespace grape {

template <typename T>
class BinaryVecOut {
 public:
  BinaryVecOut(const std::vector<T>& id_vec,
               const lcc_opt_impl::ref_vector<uint8_t>& weight_vec)
      : id_vec(id_vec), weight_vec(weight_vec) {}

  const std::vector<T>& id_vec;
  const lcc_opt_impl::ref_vector<uint8_t>& weight_vec;
};

template <typename T>
InArchive& operator<<(InArchive& arc, const BinaryVecOut<T>& vec) {
  int deg = vec.id_vec.size();
  arc << deg;
  arc.AddBytes(vec.id_vec.data(), sizeof(T) * deg);
  arc.AddBytes(vec.weight_vec.data(), sizeof(uint8_t) * deg);
  return arc;
}

template <typename T>
class BinaryVecIn {
 public:
  BinaryVecIn() {}
  ~BinaryVecIn() {}

  void reset(int deg, const T* id_ptr, const uint8_t* weight_ptr) {
    id_ptr_ = id_ptr;
    id_end_ = id_ptr + deg;
    weight_ptr_ = weight_ptr;
  }

  bool pop(T& id, uint8_t& weight) {
    if (id_ptr_ == id_end_) {
      return false;
    }
    id = *(id_ptr_++);
    weight = *(weight_ptr_++);
    return true;
  }

 private:
  const T* id_ptr_;
  const T* id_end_;
  const uint8_t* weight_ptr_;
};

template <typename T>
OutArchive& operator>>(OutArchive& arc, BinaryVecIn<T>& vec) {
  int deg;
  arc >> deg;
  const T* id_ptr = static_cast<const T*>(arc.GetBytes(sizeof(T) * deg));
  const uint8_t* weight_ptr =
      static_cast<const uint8_t*>(arc.GetBytes(sizeof(uint8_t) * deg));
  vec.reset(deg, id_ptr, weight_ptr);
  return arc;
}

// #define HASH_INTERSECT

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
class LCCDirected
    : public ParallelAppBase<FRAG_T, LCCDirectedContext<FRAG_T, COUNT_T>,
                             ParallelMessageManagerOpt>,
      public ParallelEngine {
 public:
  using fragment_t = FRAG_T;
  using context_t = LCCDirectedContext<FRAG_T, COUNT_T>;
  using message_manager_t = ParallelMessageManagerOpt;
  using worker_t = ParallelWorkerOpt<LCCDirected<FRAG_T, COUNT_T>>;
  using vid_t = typename context_t::vid_t;

  virtual ~LCCDirected() {}

  static std::shared_ptr<worker_t> CreateWorker(
      std::shared_ptr<LCCDirected<FRAG_T, COUNT_T>> app,
      std::shared_ptr<FRAG_T> frag) {
    return std::shared_ptr<worker_t>(new worker_t(app, frag));
  }
  using vertex_t = typename fragment_t::vertex_t;
  using count_t = COUNT_T;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kBothOutIn;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();

    messages.InitChannels(thread_num());

    ctx.stage = 0;

    ForEach(inner_vertices, [&frag, &messages, &ctx](int tid, vertex_t v) {
      ctx.global_degree[v] =
          frag.GetLocalOutDegree(v) + frag.GetLocalInDegree(v);
      messages.SendMsgThroughEdges<fragment_t, int>(frag, v,
                                                    ctx.global_degree[v], tid);
    });

    // Just in case we are running on single process and no messages will
    // be send. ForceContinue() ensure the computation
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    if (ctx.stage == 0) {
      ctx.stage = 1;
      messages.ParallelProcess<fragment_t, int>(
          thread_num(), frag,
          [&ctx](int tid, vertex_t u, int msg) { ctx.global_degree[u] = msg; });

      ctx.neighbor_pools.resize(thread_num());
      ctx.weight_pools.resize(thread_num());

      ForEach(inner_vertices, [&frag, &ctx, &messages](int tid, vertex_t v) {
        auto& nbr_pool = ctx.neighbor_pools[tid];
        auto& weight_pool = ctx.weight_pools[tid];
        auto& nbr_vec = ctx.complete_neighbor[v];
        auto& weight_vec = ctx.neighbor_weight[v];
        nbr_pool.reserve(ctx.global_degree[v]);
        auto oe = frag.GetOutgoingAdjList(v);
        for (auto& e : oe) {
          nbr_pool.push_back(e.get_neighbor());
        }
        auto ie = frag.GetIncomingAdjList(v);
        for (auto& e : ie) {
          nbr_pool.push_back(e.get_neighbor());
        }
        std::sort(nbr_pool.begin(), nbr_pool.end());
        size_t size = nbr_pool.size();
        weight_pool.reserve(size);
        size_t deduped_index = 0;
        for (size_t i = 0; i != size;) {
          vertex_t cur = nbr_pool[i];
          size_t j = i + 1;
          while (j != size && nbr_pool[j] == cur) {
            ++j;
          }
          nbr_pool[deduped_index++] = cur;
          weight_pool.push_back(static_cast<uint8_t>(j - i));
          i = j;
        }
        nbr_pool.resize(deduped_index);
        ctx.deduped_degree[v] = deduped_index;

        vid_t v_gid_hash = IdHasher<vid_t>::hash(frag.GetInnerVertexGid(v));
        int v_deg = ctx.global_degree[v];
        size_t filtered_index = 0;
        std::vector<vid_t> msg_vec;
        for (size_t i = 0; i < deduped_index; ++i) {
          vertex_t u = nbr_pool[i];
          if ((ctx.global_degree[u] > v_deg) ||
              ((ctx.global_degree[u] == v_deg) &&
               v_gid_hash > IdHasher<vid_t>::hash(frag.Vertex2Gid(u)))) {
            nbr_pool[filtered_index] = u;
            weight_pool[filtered_index] = weight_pool[i];
            msg_vec.push_back(frag.Vertex2Gid(u));
            ++filtered_index;
          }
        }
        nbr_pool.resize(filtered_index);
        weight_pool.resize(filtered_index);
        nbr_vec = nbr_pool.finish();
        weight_vec = weight_pool.finish();
        BinaryVecOut<vid_t> msg(msg_vec, weight_vec);
        messages.SendMsgThroughEdges<fragment_t, BinaryVecOut<vid_t>>(frag, v,
                                                                      msg, tid);
      });

      messages.ForceContinue();
    } else if (ctx.stage == 1) {
      ctx.stage = 2;
      messages.ParallelProcess<fragment_t, BinaryVecIn<vid_t>>(
          thread_num(), frag,
          [&frag, &ctx](int tid, vertex_t u, BinaryVecIn<vid_t>& msg) {
            auto& nbr_pool = ctx.neighbor_pools[tid];
            auto& weight_pool = ctx.weight_pools[tid];
            auto& nbr_vec = ctx.complete_neighbor[u];
            auto& weight_vec = ctx.neighbor_weight[u];
            vid_t gid;
            uint8_t weight;
            std::vector<std::pair<vertex_t, uint8_t>> vec;
            while (msg.pop(gid, weight)) {
              vertex_t v;
              if (frag.Gid2Vertex(gid, v)) {
                vec.emplace_back(v, weight);
              }
            }
            std::sort(vec.begin(), vec.end(),
                      [](const std::pair<vertex_t, uint8_t>& lhs,
                         const std::pair<vertex_t, uint8_t>& rhs) {
                        return lhs.first < rhs.first;
                      });
            nbr_pool.reserve(vec.size());
            weight_pool.reserve(vec.size());
            for (auto& pair : vec) {
              nbr_pool.push_back(pair.first);
              weight_pool.push_back(pair.second);
            }
            nbr_vec = nbr_pool.finish();
            weight_vec = weight_pool.finish();
          });
#ifdef HASH_INTERSECT
      std::vector<ska::flat_hash_map<vid_t, uint8_t>> weight_maps(thread_num());
#endif

#ifdef HASH_INTERSECT
      ForEach(inner_vertices, [&ctx, &weight_maps](int tid, vertex_t v) {
#else
      ForEach(inner_vertices, [&ctx](int tid, vertex_t v) {
#endif
        auto& v_nbr_vec = ctx.complete_neighbor[v];
        if (v_nbr_vec.size() <= 1) {
          return;
        }
        auto& v_nbr_weight = ctx.neighbor_weight[v];
        vid_t v_dd = v_nbr_vec.size();
        count_t v_count = 0;
#ifndef HASH_INTERSECT
        for (vid_t vi = 0; vi != v_dd; ++vi) {
          auto u = v_nbr_vec[vi];
          auto& u_nbr_vec = ctx.complete_neighbor[u];
          if (u_nbr_vec.empty()) {
            continue;
          }

          count_t uv_weight = static_cast<count_t>(v_nbr_weight[vi]);
          auto& u_nbr_weight = ctx.neighbor_weight[u];
          count_t u_count = 0;

          vid_t u_dd = u_nbr_vec.size();
          vid_t v_j = 0, u_j = 0;
          while (v_j < v_dd && u_j < u_dd) {
            if (v_nbr_vec[v_j] == u_nbr_vec[u_j]) {
              auto w = v_nbr_vec[v_j];
              u_count += static_cast<count_t>(v_nbr_weight[v_j]);
              v_count += static_cast<count_t>(u_nbr_weight[u_j]);
              atomic_add(ctx.tricnt[w], uv_weight);

              ++v_j;
              ++u_j;
            } else if (v_nbr_vec[v_j] < u_nbr_vec[u_j]) {
              ++v_j;
            } else {
              ++u_j;
            }
          }
          atomic_add(ctx.tricnt[u], u_count);
        }
#else
        auto& v_nbr_map = weight_maps[tid];
        for (vid_t vi = 0; vi != v_dd; ++vi) {
          v_nbr_map[v_nbr_vec[vi].GetValue()] = v_nbr_weight[vi];
        }
        for (vid_t vi = 0; vi != v_dd; ++vi) {
          auto u = v_nbr_vec[vi];
          auto& u_nbr_vec = ctx.complete_neighbor[u];
          if (u_nbr_vec.empty()) {
            continue;
          }
          count_t uv_weight = static_cast<count_t>(v_nbr_weight[vi]);
          auto& u_nbr_weight = ctx.neighbor_weight[u];
          count_t u_count = 0;

          vid_t u_dd = u_nbr_vec.size();
          for (vid_t ui = 0; ui != u_dd; ++ui) {
            auto w = u_nbr_vec[ui];
            auto iter = v_nbr_map.find(w.GetValue());
            if (iter != v_nbr_map.end()) {
              v_count += static_cast<count_t>(u_nbr_weight[ui]);
              u_count += static_cast<count_t>(iter->second);
              atomic_add(ctx.tricnt[w], uv_weight);
            }
          }

          atomic_add(ctx.tricnt[u], u_count);
        }
        v_nbr_map.clear();
#endif
        atomic_add(ctx.tricnt[v], v_count);
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
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_LCC_LCC_DIRECTED_H_
