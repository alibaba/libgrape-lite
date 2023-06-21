/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_WCC_WCC_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_WCC_WCC_H_
#ifdef __CUDACC__
#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class WCCContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using label_t = vid_t;

  explicit WCCContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}
#ifdef PROFILING
  ~WCCContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "BFS kernel time: " << traversal_kernel_time * 1000;
  }
#endif
  void Init(GPUMessageManager& messages, AppConfig app_config) {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto vertices = frag.Vertices();

    this->lb = app_config.lb;
    label.Init(vertices, std::numeric_limits<label_t>::max());
    label.H2D();

    auto in_cap = app_config.wl_alloc_factor_in * frag.GetEdgeNum();

    in_q.Init(vertices);
    out_q_local.Init(vertices);
    out_q_remote.Init(vertices);
    tmp_q.Init(iv.size());

    // <gid, gid> will be packed and sent to destinations
    messages.InitBuffer(ov.size() * (sizeof(vid_t) + sizeof(label_t)),
                        in_cap * sizeof(vid_t) * 2);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    label.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << label[v] << std::endl;
    }
  }

  LoadBalancing lb{};
  VertexArray<label_t, vid_t> label;
  DenseVertexSet<vid_t> in_q, out_q_local, out_q_remote;
  Queue<vertex_t> tmp_q;
  double get_msg_time{};
  double traversal_kernel_time{};
};

template <typename FRAG_T>
class WCC : public GPUAppBase<FRAG_T, WCCContext<FRAG_T>>,
            public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(WCC<FRAG_T>, WCCContext<FRAG_T>, FRAG_T)
  using label_t = typename context_t::label_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;
  static constexpr bool need_build_device_vm = true;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto av = frag.Vertices();
    WorkSourceRange<vertex_t> ws_in(*av.begin(), av.size());

    LaunchKernel(
        messages.stream(),
        [=] __device__(dev_fragment_t d_frag,
                       dev::VertexArray<label_t, vid_t> label,
                       dev::DenseVertexSet<vid_t> d_in_q) mutable {
          auto tid = TID_1D;
          auto total_nthreads = TOTAL_THREADS_1D;
          auto size = ws_in.size();

          for (size_t idx = 0 + tid; idx < size; idx += total_nthreads) {
            vertex_t v = ws_in.GetWork(idx);

            // label[v] = d_frag.GetId(v);
            label[v] = d_frag.Vertex2Gid(v);
            d_in_q.Insert(v);
          }
        },
        frag.DeviceObject(), ctx.label.DeviceObject(), ctx.in_q.DeviceObject());
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_label = ctx.label.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local = ctx.out_q_local;
    auto d_out_q_local = out_q_local.DeviceObject();
    auto& out_q_remote = ctx.out_q_remote;
    auto d_out_q_remote = out_q_remote.DeviceObject();
    auto& tmp_q = ctx.tmp_q;
    auto d_tmp_q = tmp_q.DeviceObject();
    auto d_frag = frag.DeviceObject();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto& stream = messages.stream();
    auto d_mm = messages.DeviceObject();

#ifdef PROFILING
    ctx.get_msg_time -= grape::GetCurrentTime();
#endif
    messages.template ParallelProcess<dev_fragment_t, label_t>(
        d_frag, [=] __device__(vertex_t v, label_t received_gid) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (received_gid < atomicMin(&d_label[v], received_gid)) {
            d_in_q.Insert(v);
          }
        });
#ifdef PROFILING
    auto in_size = in_q.Count(stream);
    VLOG(1) << "Frag " << frag.fid() << " In: " << in_size;
    ctx.get_msg_time += grape::GetCurrentTime();
    auto traversal_kernel_time = grape::GetCurrentTime();
#endif

    {
      WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());

      tmp_q.Clear(stream);
      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_in_q.Exist(v)) {
          d_tmp_q.AppendWarp(v);
        }
      });
    }
    WorkSourceArray<vertex_t> ws_in(tmp_q.data(), tmp_q.size(stream));

    ForEachOutgoingEdge(
        stream, d_frag, ws_in,
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          label_t u_label = d_label[u];
          vertex_t v = nbr.get_neighbor();

          if (u_label < atomicMin(&d_label[v], u_label)) {
            if (d_frag.IsInnerVertex(v)) {
              d_out_q_local.Insert(v);
            } else {
              d_out_q_remote.Insert(v);
            }
          }
        },
        ctx.lb);

    if (frag.load_strategy == grape::LoadStrategy::kBothOutIn) {
      ForEachIncomingEdge(
          stream, d_frag, ws_in,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            label_t u_label = d_label[u];
            vertex_t v = nbr.get_neighbor();

            if (u_label < atomicMin(&d_label[v], u_label)) {
              if (d_frag.IsInnerVertex(v)) {
                d_out_q_local.Insert(v);
              } else {
                d_out_q_remote.Insert(v);
              }
            }
          },
          ctx.lb);
    }

    auto ws_ov = WorkSourceRange<vertex_t>(*ov.begin(), ov.size());

    ForEach(stream, ws_ov, [=] __device__(vertex_t v) mutable {
      if (d_out_q_remote.Exist(v)) {
        d_mm.template SyncStateOnOuterVertexWarpOpt(d_frag, v, d_label[v]);
      }
    });
    auto local_size = out_q_local.Count(stream);
#ifdef PROFILING
    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    VLOG(2) << "Frag " << frag.fid() << " Local out: " << local_size
            << " Kernel time: " << traversal_kernel_time * 1000;
    ctx.traversal_kernel_time += traversal_kernel_time;
#endif
    in_q.Clear(stream);
    out_q_remote.Clear(stream);
    out_q_local.Swap(in_q);
    if (local_size > 0) {
      messages.ForceContinue();
    }
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_WCC_WCC_H_
