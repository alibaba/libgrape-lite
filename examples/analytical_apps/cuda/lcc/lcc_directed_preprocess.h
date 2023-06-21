/** Copyright 2023 Alibaba Group Holding Limited.

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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_PREPROCESS_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_PREPROCESS_H_

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "cuda/app_config.h"
#include "grape/grape.h"

#define LCC_M 16
#define LCC_CHUNK_SIZE(I, N, m) (((I) < ((N) % (m))) + (N) / (m))
#define LCC_CHUNK_START(I, N, m) \
  (((I) < ((N) % (m)) ? (I) : ((N) % (m))) + (I) * ((N) / (m)))

namespace grape {

template <typename FRAG_T>
class LCCDPContext : public grape::VoidContext<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;

  explicit LCCDPContext(const FRAG_T& fragment)
      : grape::VoidContext<FRAG_T>(fragment) {}

  void Init(ParallelMessageManager& messages, grape::cuda::AppConfig app_config,
            vid_t** sorted_col, size_t** row_offset, char** weight,
            size_t** true_degree) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    global_degree.Init(vertices);
    true_global_degree.Init(frag.InnerVertices());
    complete_neighbor.Init(vertices);
    complete_neighbor_weight.Init(vertices);
    this->sorted_col = sorted_col;
    this->row_offset = row_offset;
    this->true_degree = true_degree;
    this->weight = weight;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();
    *true_degree =
        reinterpret_cast<size_t*>(malloc(iv.size() * sizeof(size_t)));
    memset(*true_degree, 0, sizeof(size_t) * iv.size());
    for (auto v : iv) {
      (*true_degree)[v.GetValue()] = true_global_degree[v];
    }

    auto vertices = frag.Vertices();
    size_t size = vertices.size();
    *row_offset =
        reinterpret_cast<size_t*>(malloc((size + 1) * sizeof(size_t)));
    memset(*row_offset, 0, sizeof(size_t) * (size + 1));
    for (auto v : vertices) {
      auto& nbrs = complete_neighbor[v];
      auto degree = global_degree[v];
      int idx = v.GetValue();
      (*row_offset)[idx + 1] = (*row_offset)[idx] + degree;
    }
    size_t edge_size = (*row_offset)[size];
    *sorted_col = reinterpret_cast<vid_t*>(malloc(edge_size * sizeof(vid_t)));
    *weight = reinterpret_cast<char*>(malloc(edge_size * sizeof(char)));
    for (auto v : vertices) {
      auto& nbrs = complete_neighbor[v];
      auto& nbr_weight = complete_neighbor_weight[v];
      auto degree = global_degree[v];
      size_t base = (*row_offset)[v.GetValue()];
      for (size_t i = 0; i < degree; ++i) {
        assert(nbr_weight[i] <= 2);
        (*sorted_col)[base + i] = nbrs[i].GetValue();
        (*weight)[base + i] = nbr_weight[i];
      }
    }
  }

  typename FRAG_T::template vertex_array_t<uint32_t> global_degree;
  typename FRAG_T::template vertex_array_t<uint32_t> true_global_degree;
  typename FRAG_T::template vertex_array_t<std::vector<vertex_t>>
      complete_neighbor;
  typename FRAG_T::template vertex_array_t<std::vector<char>>
      complete_neighbor_weight;
  int stage = 0;
  vid_t** sorted_col;
  size_t** row_offset;
  size_t** true_degree;
  char** weight;
};

template <typename FRAG_T>
class LCCDP : public ParallelAppBase<FRAG_T, LCCDPContext<FRAG_T>>,
              public ParallelEngine {
 public:
  INSTALL_PARALLEL_WORKER(LCCDP<FRAG_T>, LCCDPContext<FRAG_T>, FRAG_T);
  using vertex_t = typename fragment_t::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kBothOutIn;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();

    messages.InitChannels(thread_num());

    ctx.stage = 0;

    // Each vertex scatter its own out and in degree.
    ForEach(inner_vertices, [&messages, &frag, &ctx](int tid, vertex_t v) {
      ctx.global_degree[v] =
          frag.GetLocalOutDegree(v) + frag.GetLocalInDegree(v);
      messages.SendMsgThroughEdges<fragment_t, uint32_t>(
          frag, v, ctx.global_degree[v], tid);

      auto& nbr_vec = ctx.complete_neighbor[v];
      nbr_vec.reserve(ctx.global_degree[v]);
      auto oes = frag.GetOutgoingAdjList(v);
      auto ies = frag.GetIncomingAdjList(v);

      for (auto e = oes.begin(); e != oes.begin() + oes.Size(); e++) {
        auto u = e->get_neighbor();
        nbr_vec.push_back(u);
      }
      for (auto e = ies.begin(); e != ies.begin() + ies.Size(); e++) {
        auto u = e->get_neighbor();
        nbr_vec.push_back(u);
      }
      std::sort(nbr_vec.begin(), nbr_vec.end());
      ctx.true_global_degree[v] =
          std::unique(nbr_vec.begin(), nbr_vec.end()) - nbr_vec.begin();
      nbr_vec.clear();
      nbr_vec.shrink_to_fit();
    });

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    using vid_t = typename context_t::vid_t;

    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();

    if (ctx.stage == 0) {
      messages.ParallelProcess<fragment_t, uint32_t>(
          thread_num(), frag, [&ctx](int tid, vertex_t u, uint32_t msg) {
            ctx.global_degree[u] = msg;
          });
    }

    if (ctx.stage >= 0 && ctx.stage <= LCC_M) {
      int K = ctx.stage;

      if (K > 0) {
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
      }

      if (K < LCC_M) {
        ForEach(inner_vertices,
                [this, &frag, &ctx, &messages, K](int tid, vertex_t v) {
                  vid_t u_gid, v_gid;
                  auto& nbr_vec = ctx.complete_neighbor[v];
                  uint32_t degree = ctx.global_degree[v];
                  nbr_vec.reserve(degree);
                  std::vector<vid_t> msg_vec;
                  msg_vec.reserve(degree);
                  auto oes = frag.GetOutgoingAdjList(v);
                  auto ies = frag.GetIncomingAdjList(v);

                  size_t o_length = oes.Size();
                  size_t o_chunk_start = LCC_CHUNK_START(K, o_length, LCC_M);
                  size_t o_chunk_end =
                      o_chunk_start + LCC_CHUNK_SIZE(K, o_length, LCC_M);
                  for (auto e = oes.begin() + o_chunk_start;
                       e != oes.begin() + o_chunk_end; e++) {
                    auto u = e->get_neighbor();
                    if (ctx.global_degree[u] > ctx.global_degree[v]) {
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

                  size_t i_length = ies.Size();
                  size_t i_chunk_start = LCC_CHUNK_START(K, i_length, LCC_M);
                  size_t i_chunk_end =
                      i_chunk_start + LCC_CHUNK_SIZE(K, i_length, LCC_M);
                  for (auto e = ies.begin() + i_chunk_start;
                       e != ies.begin() + i_chunk_end; e++) {
                    auto u = e->get_neighbor();
                    if (ctx.global_degree[u] > ctx.global_degree[v]) {
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
                  messages.SendMsgThroughEdges<fragment_t, std::vector<vid_t>>(
                      frag, v, msg_vec, tid);
                });
      }
      messages.ForceContinue();
    }

    if (ctx.stage == LCC_M) {
      auto vertices = frag.Vertices();
      ForEach(vertices, [this, &ctx, &frag](int tid, vertex_t v) {
        auto& nbr_vec = ctx.complete_neighbor[v];
        auto& nbr_weight = ctx.complete_neighbor_weight[v];
        std::sort(nbr_vec.begin(), nbr_vec.end());
        nbr_weight.resize(nbr_vec.size(), 1);
        size_t i = 1;
        for (size_t j = 1; j < nbr_vec.size(); ++j) {
          if (nbr_vec[i - 1] == nbr_vec[j]) {
            nbr_weight[i - 1]++;
            continue;
          }
          nbr_vec[i] = nbr_vec[j];
          i++;
        }
        ctx.global_degree[v] = nbr_vec.size() == 0 ? 0 : i;
      });
    }
    ctx.stage = ctx.stage + 1;
  }
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_PREPROCESS_H_
