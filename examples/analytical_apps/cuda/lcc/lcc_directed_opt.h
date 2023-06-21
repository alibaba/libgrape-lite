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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_OPT_H_
#ifdef __CUDACC__
#include <iomanip>
#include <iostream>

#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class LCCDOPTContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using msg_t = vid_t;

  explicit LCCDOPTContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config,
            msg_t** sorted_col, size_t** offset, char** weight,
            size_t** true_degree) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->lb = app_config.lb;
    this->stage = 0;

    global_degree.Init(vertices, 0);
    tricnt.Init(vertices, 0);
    hi_q.Init(iv.size());
    lo_q.Init(iv.size());
    tricnt.H2D();

    row_offset.resize(vertices.size() + 1, 0);
    size_t n_vertices = vertices.size();

    auto d_global_degree = global_degree.DeviceObject();
    CHECK_CUDA(cudaMemcpy(d_global_degree.data(), *true_degree,
                          sizeof(size_t) * iv.size(), cudaMemcpyHostToDevice));

    auto* d_row_offset = thrust::raw_pointer_cast(row_offset.data());
    CHECK_CUDA(cudaMemcpy(d_row_offset, *offset,
                          sizeof(size_t) * (n_vertices + 1),
                          cudaMemcpyHostToDevice));

    size_t n_edges = (*offset)[n_vertices];
    col_indices.resize(n_edges, 0);
    col_indices_weight.resize(n_edges, 0);
    auto* d_col_indices = thrust::raw_pointer_cast(col_indices.data());
    auto* d_col_indices_weight =
        thrust::raw_pointer_cast(col_indices_weight.data());
    CHECK_CUDA(cudaMemcpy(d_col_indices, *sorted_col, sizeof(msg_t) * n_edges,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices_weight, *weight, sizeof(char) * n_edges,
                          cudaMemcpyHostToDevice));

    messages.InitBuffer(
        ov.size() * (sizeof(thrust::pair<vid_t, size_t>)),
        1 * (sizeof(thrust::pair<vid_t, msg_t>)));  // rely on syncLengths()
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    tricnt.D2H();
    global_degree.D2H();

    for (auto v : iv) {
      double score = 0;
      if (global_degree[v] >= 2) {
        score = 1.0 * (tricnt[v]) /
                (static_cast<int64_t>(global_degree[v]) *
                 (static_cast<int64_t>(global_degree[v]) - 1));
      }
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << score << std::endl;
    }
  }

  LoadBalancing lb{};
  VertexArray<size_t, vid_t> global_degree;
  VertexArray<size_t, vid_t> tricnt;
  Queue<vertex_t, vid_t> hi_q, lo_q;
  thrust::device_vector<size_t> row_offset;
  thrust::device_vector<msg_t> col_indices;
  thrust::device_vector<char> col_indices_weight;
  int stage{};
};

template <typename FRAG_T>
class LCCDOPT : public GPUAppBase<FRAG_T, LCCDOPTContext<FRAG_T>>,
                public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(LCCDOPT<FRAG_T>, LCCDOPTContext<FRAG_T>, FRAG_T)
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;
  using msg_t = vid_t;

  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;
  static constexpr bool need_build_device_vm = true;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto d_frag = frag.DeviceObject();
    auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
    auto d_mm = messages.DeviceObject();
    WorkSourceRange<vertex_t> ws_in(*inner_vertices.begin(),
                                    inner_vertices.size());
    messages.ForceContinue();
  }

  void TriangleCounting(const fragment_t& frag, context_t& ctx,
                        message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto d_tricnt = ctx.tricnt.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& lo_q = ctx.lo_q;
    auto& hi_q = ctx.hi_q;
    auto d_lo_q = lo_q.DeviceObject();
    auto d_hi_q = hi_q.DeviceObject();

    {
      WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      ForEach(stream, ws_in, [=] __device__(vertex_t u) mutable {
        size_t idx = u.GetValue();
        size_t degree = d_row_offset[idx + 1] - d_row_offset[idx];
        if (degree > 1024) {
          d_hi_q.Append(u);
        } else {
          d_lo_q.Append(u);
        }
      });
    }
    stream.Sync();
    {
      WorkSourceArray<vertex_t> ws_in(lo_q.data(), lo_q.size(stream));
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = d_row_offset + 1;
      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
      auto* d_col_indices_weight =
          thrust::raw_pointer_cast(ctx.col_indices_weight.data());
      ForEachWithIndexWarpDynamic(
          stream, ws_in,
          [=] __device__(size_t lane, size_t idx, vertex_t u) mutable {
            idx = u.GetValue();
            size_t triangle_count = 0;
            for (auto eid = d_row_offset[idx]; eid < d_filling_offset[idx];
                 eid++) {
              vertex_t v(d_col_indices[eid]);
              auto edge_begin_u = d_row_offset[u.GetValue()],
                   edge_end_u = d_filling_offset[u.GetValue()];
              auto edge_begin_v = d_row_offset[v.GetValue()],
                   edge_end_v = d_filling_offset[v.GetValue()];
              size_t degree_u = edge_end_u - edge_begin_u;
              size_t degree_v = edge_end_v - edge_begin_v;
              int w_weight = d_col_indices_weight[eid];
              size_t u_cnt = 0;
              size_t v_cnt = 0;
              dev::intersect_num_d(&d_col_indices[edge_begin_u], degree_u,
                                   &d_col_indices[edge_begin_v], degree_v,
                                   &d_col_indices_weight[edge_begin_u], &u_cnt,
                                   &d_col_indices_weight[edge_begin_v], &v_cnt,
                                   [=] __device__(msg_t key) mutable {
                                     dev::atomicAdd64(&d_tricnt[vertex_t(key)],
                                                      w_weight);
                                   });
              if (lane == 0) {
                dev::atomicAdd64(&d_tricnt[v], v_cnt);
                triangle_count += u_cnt;
              }
            }
            if (lane == 0 && triangle_count != 0) {
              dev::atomicAdd64(&d_tricnt[u], triangle_count);
            }
          });
    }
    {
      WorkSourceArray<vertex_t> ws_in(hi_q.data(), hi_q.size(stream));
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = d_row_offset + 1;
      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
      auto* d_col_indices_weight =
          thrust::raw_pointer_cast(ctx.col_indices_weight.data());
      ForEachWithIndexBlockDynamic(
          stream, ws_in,
          [=] __device__(size_t lane, size_t idx, vertex_t u) mutable {
            idx = u.GetValue();
            size_t triangle_count = 0;
            for (auto eid = d_row_offset[idx]; eid < d_filling_offset[idx];
                 eid++) {
              vertex_t v(d_col_indices[eid]);
              auto edge_begin_u = d_row_offset[u.GetValue()],
                   edge_end_u = d_filling_offset[u.GetValue()];
              auto edge_begin_v = d_row_offset[v.GetValue()],
                   edge_end_v = d_filling_offset[v.GetValue()];
              size_t degree_u = edge_end_u - edge_begin_u;
              size_t degree_v = edge_end_v - edge_begin_v;
              int w_weight = d_col_indices_weight[eid];
              size_t u_cnt = 0;
              size_t v_cnt = 0;
              dev::intersect_num_blk_d(
                  &d_col_indices[edge_begin_u], degree_u,
                  &d_col_indices[edge_begin_v], degree_v,
                  &d_col_indices_weight[edge_begin_u], &u_cnt,
                  &d_col_indices_weight[edge_begin_v], &v_cnt,
                  [=] __device__(msg_t key) mutable {
                    dev::atomicAdd64(&d_tricnt[vertex_t(key)], w_weight);
                  });
              if (lane == 0) {
                dev::atomicAdd64(&d_tricnt[v], v_cnt);
                triangle_count += u_cnt;
              }
              __syncthreads();
            }
            __syncthreads();
            if (lane == 0) {
              dev::atomicAdd64(&d_tricnt[u], triangle_count);
            }
          });
    }

    {
      WorkSourceRange<vertex_t> ws_in(*ov.begin(), ov.size());
      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_tricnt[v] != 0) {
          d_mm.template SyncStateOnOuterVertex(d_frag, v, d_tricnt[v]);
        }
      });
    }
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    if (ctx.stage == 0) {
      TriangleCounting(frag, ctx, messages);
    }

    if (ctx.stage == 1) {
      auto d_frag = frag.DeviceObject();
      auto d_tricnt = ctx.tricnt.DeviceObject();
      messages.template ParallelProcess<dev_fragment_t, size_t>(
          d_frag, [=] __device__(vertex_t v, size_t tri_cnt) mutable {
            dev::atomicAdd64(&d_tricnt[v], tri_cnt);
          });
    }

    ctx.stage = ctx.stage + 1;
  }
};

}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_OPT_H_
