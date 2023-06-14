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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_H_
#ifdef __CUDACC__
#include <iomanip>
#include <iostream>

#include "cuda/app_config.h"
#include "grape/grape.h"

#define LCCD_M 24
#define LCCD_CHUNK_SIZE(I, N, m) (((I) < ((N) % (m))) + (N) / (m))
#define LCCD_CHUNK_START(I, N, m) \
  (((I) < ((N) % (m)) ? (I) : ((N) % (m))) + (I) * ((N) / (m)))

namespace grape {
namespace cuda {
template <typename FRAG_T>
class LCCDContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using msg_t = vid_t;

  explicit LCCDContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~LCCDContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "Pagerank kernel time: " << traversal_kernel_time * 1000;
    VLOG(1) << "Send msg time: " << send_msg_time * 1000;
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    this->lb = app_config.lb;
    this->stage = 0;

    global_degree.Init(vertices, 0);
    filling_offset.Init(vertices, 0);
    tricnt.Init(vertices, 0);
    tricnt.H2D();

    row_offset.resize(vertices.size() + 1, 0);
    compact_row_offset.resize(vertices.size() + 1, 0);

    size_t n_edges = 0;
    size_t n_vertices = 0;
    using nbr_t = typename FRAG_T::nbr_t;
    for (auto u : frag.InnerVertices()) {
      n_edges += frag.GetLocalOutDegree(u) + frag.GetLocalInDegree(u);
      n_vertices += 1;
    }

    messages.InitBuffer(
        std::max(
            (n_edges / LCCD_M + 1) * 2 * (sizeof(thrust::pair<vid_t, msg_t>)),
            n_vertices * (sizeof(size_t))),
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
        score = (tricnt[v]) / (2.0 * static_cast<int64_t>(global_degree[v]) *
                               (static_cast<int64_t>(global_degree[v]) - 1));
      }
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << score << std::endl;
    }
  }

  LoadBalancing lb{};
  VertexArray<size_t, vid_t> global_degree;
  VertexArray<size_t, vid_t> filling_offset;
  VertexArray<size_t, vid_t> tricnt;
  thrust::device_vector<size_t> row_offset;
  thrust::device_vector<size_t> compact_row_offset;
  thrust::device_vector<msg_t> col_indices;
  thrust::device_vector<msg_t> col_sorted_indices;
  thrust::device_vector<char> col_dup_indices;
  thrust::device_vector<char> col_double_indices;
  int stage{};
#ifdef PROFILING
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
#endif
};

template <typename FRAG_T>
class LCCD : public GPUAppBase<FRAG_T, LCCDContext<FRAG_T>>,
             public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(LCCD<FRAG_T>, LCCDContext<FRAG_T>, FRAG_T)
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
    auto d_global_degree = ctx.global_degree.DeviceObject();
    auto d_mm = messages.DeviceObject();
    WorkSourceRange<vertex_t> ws_in(*inner_vertices.begin(),
                                    inner_vertices.size());

    ForEach(messages.stream(), ws_in, [=] __device__(vertex_t v) mutable {
      msg_t degree = d_frag.GetLocalOutDegree(v) + d_frag.GetLocalInDegree(v);
      d_global_degree[v] = degree;
      d_mm.template SendMsgThroughEdges(d_frag, v, degree);
    });
    messages.ForceContinue();
  }

  void PopulateAdjList(const fragment_t& frag, context_t& ctx,
                       message_manager_t& messages) {
    auto dev_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto d_global_degree = ctx.global_degree.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
    auto d_filling_offset = ctx.filling_offset.DeviceObject();
    // Recieve outer degrees.
    messages.template ParallelProcess<dev_fragment_t, msg_t>(
        dev_frag, [=] __device__(vertex_t v, msg_t degree) mutable {
          d_global_degree[v] = degree;
        });

    InclusiveSum(d_global_degree.data(), d_row_offset + 1, vertices.size(),
                 stream.cuda_stream());

    CHECK_CUDA(cudaMemcpyAsync(d_filling_offset.data(), d_row_offset,
                               sizeof(size_t) * vertices.size(),
                               cudaMemcpyDeviceToDevice, stream.cuda_stream()));

    stream.Sync();
    auto n_filtered_edges = ctx.row_offset[vertices.size()];
    ctx.col_indices.resize(n_filtered_edges);
    ctx.col_sorted_indices.resize(n_filtered_edges);
    ctx.col_dup_indices.resize(n_filtered_edges);

    auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());

    // filling edges with precomputed gids
    ForEachOutgoingEdge(
        stream, dev_frag, ws_in,
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          auto pos = dev::atomicAdd64(&d_filling_offset[u], 1);
          d_col_indices[pos] = v.GetValue();
        },
        ctx.lb);
    ForEachIncomingEdge(
        stream, dev_frag, ws_in,
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          auto pos = dev::atomicAdd64(&d_filling_offset[u], 1);
          d_col_indices[pos] = v.GetValue();
        },
        ctx.lb);
  }

  void TransferAdjList(const fragment_t& frag, context_t& ctx,
                       message_manager_t& messages) {
    int K = ctx.stage;
    auto dev_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_mm = messages.DeviceObject();
    auto d_filling_offset = ctx.filling_offset.DeviceObject();
    auto* d_row_offset_hi = d_filling_offset.data();
    auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
    auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());

    if (K > 0) {
      messages.template ParallelProcess<dev_fragment_t, msg_t>(
          dev_frag, [=] __device__(vertex_t u, msg_t v_gid) mutable {
            vertex_t v;
            assert(dev_frag.IsOuterVertex(u));
            if (dev_frag.Gid2Vertex(v_gid, v)) {
              auto pos = dev::atomicAdd64(&d_filling_offset[u], 1);
              assert(pos + 1 <= d_row_offset[u.GetValue() + 1]);
              d_col_indices[pos] = v.GetValue();
            }
          });
    }

    if (K < LCCD_M) {
      WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
      ForEachWithIndex(
          stream, ws_in, [=] __device__(uint32_t idx, vertex_t u) mutable {
            // TODO(mengke): replace it with ForEachOutgoingEdge
            size_t length = (d_row_offset_hi[idx] - d_row_offset[idx]);
            size_t chunk_start =
                d_row_offset[idx] + LCCD_CHUNK_START(K, length, LCCD_M);
            size_t chunk_end = chunk_start + LCCD_CHUNK_SIZE(K, length, LCCD_M);

            for (auto begin = chunk_start; begin < chunk_end; begin++) {
              msg_t v_gid = dev_frag.Vertex2Gid(vertex_t(d_col_indices[begin]));
              d_mm.template SendMsgThroughEdges(dev_frag, u, v_gid);
            }
          });
      stream.Sync();
      messages.ForceContinue();
    }
  }

  size_t CompressAdjList(const fragment_t& frag, context_t& ctx,
                         message_manager_t& messages, msg_t*& sorted_col) {
    auto dev_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto vertices = frag.Vertices();
    auto d_mm = messages.DeviceObject();
    auto size = vertices.size();
    size_t valid_esize = 0;
    msg_t* buffer_col;

    // Make space;
    frag.OffloadTopology();
    {  // cacluate valid edges
      auto d_global_degree = ctx.global_degree.DeviceObject();
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = ctx.filling_offset.DeviceObject().data();
      WorkSourceRange<vertex_t> ws_in(*vertices.begin(), vertices.size());
      ForEachWithIndex(
          stream, ws_in, [=] __device__(uint32_t idx, vertex_t u) mutable {
            // TODO(mengke): replace it with ForEachOutgoingEdge
            size_t length = (d_filling_offset[idx] - d_row_offset[idx]);
            assert(d_filling_offset[idx] >= d_row_offset[idx] &&
                   d_filling_offset[idx] <= d_row_offset[idx + 1]);
            d_global_degree[u] = length;
          });
    }

    {
      WorkSourceRange<vertex_t> ws_in(*vertices.begin(), vertices.size());
      size_t n_vertices = vertices.size();
      size_t num_items = ctx.col_indices.size();  // n_edges;
      size_t num_segments = n_vertices;
      auto* d_offsets = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = ctx.filling_offset.DeviceObject().data();
      auto* d_keys_in = thrust::raw_pointer_cast(ctx.col_indices.data());
      auto* d_keys_out =
          thrust::raw_pointer_cast(ctx.col_sorted_indices.data());

      stream.Sync();
      sorted_col = SegmentSort(d_keys_in, d_keys_out, d_offsets,
                               d_filling_offset, num_items, num_segments);
      if (sorted_col == d_keys_in) {
        buffer_col = d_keys_out;
      } else if (sorted_col == d_keys_out) {
        buffer_col = d_keys_in;
      } else {
        assert(false);
      }
    }

    {
      auto* d_offsets = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = ctx.filling_offset.DeviceObject().data();
      auto* d_dup = thrust::raw_pointer_cast(ctx.col_dup_indices.data());
      auto d_global_degree = ctx.global_degree.DeviceObject();
      WorkSourceRange<vertex_t> ws_in(*vertices.begin(), vertices.size());
      ForEachWithIndex(stream, ws_in,
                       [=] __device__(uint32_t idx, vertex_t u) mutable {
                         for (auto begin = d_offsets[idx] + 1;
                              begin < d_filling_offset[idx]; begin++) {
                           auto pre = sorted_col[begin - 1];
                           auto cur = sorted_col[begin];
                           if (pre == cur) {
                             d_dup[begin - 1] = 1;  // mark as double
                             d_dup[begin] = 2;      // mark as deleted
                             d_global_degree[u] -= 1;
                           }
                         }
                       });
    }

    {  // compact col index
      auto d_global_degree = ctx.global_degree.DeviceObject();
      auto* d_compact_row_offset =
          thrust::raw_pointer_cast(ctx.compact_row_offset.data());

      InclusiveSum(d_global_degree.data(), d_compact_row_offset + 1,
                   vertices.size(), stream.cuda_stream());
      stream.Sync();
      valid_esize = ctx.compact_row_offset[vertices.size()];

      ctx.col_double_indices.resize(valid_esize);

      auto* d_offsets = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = ctx.filling_offset.DeviceObject().data();
      auto* d_compact_offset =
          thrust::raw_pointer_cast(ctx.compact_row_offset.data());
      auto* d_keys_in = sorted_col;
      auto* d_keys_out = buffer_col;
      auto* d_dup = thrust::raw_pointer_cast(ctx.col_dup_indices.data());
      auto* d_double = thrust::raw_pointer_cast(ctx.col_double_indices.data());
      WorkSourceRange<vertex_t> ws_in(*vertices.begin(), vertices.size());
      ForEachWithIndex(stream, ws_in,
                       [=] __device__(uint32_t idx, vertex_t u) mutable {
                         // TODO(mengke): replace it with ForEachOutgoingEdge
                         size_t tmp = d_compact_offset[idx];
                         for (auto begin = d_offsets[idx];
                              begin < d_filling_offset[idx]; begin++) {
                           if (d_dup[begin] != 2) {
                             d_keys_out[tmp] = d_keys_in[begin];
                             d_double[tmp] = d_dup[begin] + 1;  // 0->1 ; 1->2;
                             tmp++;
                           }
                         }
                         assert(tmp <= d_compact_offset[idx + 1]);
                       });
      stream.Sync();
      sorted_col = buffer_col;
    }
    return valid_esize;
  }

  void TriangleCounting(const fragment_t& frag, context_t& ctx,
                        message_manager_t& messages, msg_t*& sorted_col,
                        size_t valid_esize) {
    auto dev_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto d_tricnt = ctx.tricnt.DeviceObject();
    auto d_mm = messages.DeviceObject();

    {
      WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
      auto* d_row_offset =
          thrust::raw_pointer_cast(ctx.compact_row_offset.data());
      auto* d_double = thrust::raw_pointer_cast(ctx.col_double_indices.data());
      auto* d_filling_offset = d_row_offset + 1;
      auto* d_col_indices = sorted_col;
      ForEachWithIndexWarp(
          stream, ws_in,
          [=] __device__(size_t lane, size_t idx, vertex_t u) mutable {
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
              size_t num = dev::intersect_num_directed(
                  &d_col_indices[edge_begin_u], degree_u,
                  &d_col_indices[edge_begin_v], degree_v,
                  &d_double[edge_begin_v]);
              if (lane == 0) {
                triangle_count += num;
              }
            }
            if (lane == 0) {
              dev::atomicAdd64(&d_tricnt[u], triangle_count);
            }
          });
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    if (ctx.stage == 0) {
      PopulateAdjList(frag, ctx, messages);
    }

    if (ctx.stage >= 0 && ctx.stage <= LCCD_M) {
      TransferAdjList(frag, ctx, messages);
    }

    if (ctx.stage == LCCD_M) {
      msg_t* sorted_col = NULL;
      size_t valid_esize = CompressAdjList(frag, ctx, messages, sorted_col);
      // Sort destinations with segmented sort
      TriangleCounting(frag, ctx, messages, sorted_col, valid_esize);
    }

    ctx.stage = ctx.stage + 1;
  }
};

}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_DIRECTED_H_
