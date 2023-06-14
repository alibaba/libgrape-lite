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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_PREPROCESS_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_PREPROCESS_H_

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
class LCCPContext : public grape::VoidContext<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;

  explicit LCCPContext(const FRAG_T& fragment)
      : grape::VoidContext<FRAG_T>(fragment) {}

  void Init(ParallelMessageManager& messages, grape::cuda::AppConfig app_config,
            vid_t** sorted_col, size_t** row_offset) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    global_degree.Init(vertices);
    complete_neighbor.Init(vertices);
    this->sorted_col = sorted_col;
    this->row_offset = row_offset;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    size_t size = vertices.size();
    *row_offset =
        reinterpret_cast<size_t*>(malloc((size + 1) * sizeof(size_t)));
    memset(*row_offset, 0, sizeof(size_t) * (size + 1));
    for (auto v : vertices) {
      auto& nbrs = complete_neighbor[v];
      int idx = v.GetValue();
      (*row_offset)[idx + 1] = (*row_offset)[idx] + nbrs.size();
    }
    size_t edge_size = (*row_offset)[size];
    *sorted_col = reinterpret_cast<vid_t*>(malloc(edge_size * sizeof(vid_t)));
    for (auto v : vertices) {
      auto& nbrs = complete_neighbor[v];
      size_t base = (*row_offset)[v.GetValue()];
      for (size_t i = 0; i < nbrs.size(); ++i) {
        (*sorted_col)[base + i] = nbrs[i].GetValue();
      }
    }
  }

  typename FRAG_T::template vertex_array_t<uint32_t> global_degree;
  typename FRAG_T::template vertex_array_t<std::vector<vertex_t>>
      complete_neighbor;
  int stage = 0;
  vid_t** sorted_col;
  size_t** row_offset;
};

template <typename FRAG_T>
class LCCP : public ParallelAppBase<FRAG_T, LCCPContext<FRAG_T>>,
             public ParallelEngine {
 public:
  INSTALL_PARALLEL_WORKER(LCCP<FRAG_T>, LCCPContext<FRAG_T>, FRAG_T);
  using vertex_t = typename fragment_t::vertex_t;

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
      messages.SendMsgThroughOEdges<fragment_t, uint32_t>(
          frag, v, ctx.global_degree[v], tid);
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
        ForEach(inner_vertices, [this, &frag, &ctx, &messages, K](int tid,
                                                                  vertex_t v) {
          vid_t u_gid, v_gid;
          auto& nbr_vec = ctx.complete_neighbor[v];
          uint32_t degree = ctx.global_degree[v];
          nbr_vec.reserve(degree);
          auto es = frag.GetOutgoingAdjList(v);
          std::vector<vid_t> msg_vec;
          msg_vec.reserve(degree);
          size_t length = es.Size();
          size_t chunk_start = LCC_CHUNK_START(K, length, LCC_M);
          size_t chunk_end = chunk_start + LCC_CHUNK_SIZE(K, length, LCC_M);

          for (auto e = es.begin() + chunk_start; e != es.begin() + chunk_end;
               e++) {
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
          messages.SendMsgThroughOEdges<fragment_t, std::vector<vid_t>>(
              frag, v, msg_vec, tid);
        });
      }
      messages.ForceContinue();
    }

    if (ctx.stage == LCC_M) {
      auto vertices = frag.Vertices();
      ForEach(vertices, [this, &ctx](int tid, vertex_t v) {
        auto& nbr_vec = ctx.complete_neighbor[v];
        std::sort(nbr_vec.begin(), nbr_vec.end());
      });
    }
    ctx.stage = ctx.stage + 1;
  }
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_PREPROCESS_H_
