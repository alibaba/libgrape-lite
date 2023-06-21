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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_SSSP_SSSP_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_SSSP_SSSP_H_
#ifdef __CUDACC__
#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class SSSPContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
#ifdef INT_WEIGHT
  using dist_t = uint32_t;
#else
  using dist_t = float;
#endif
  explicit SSSPContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~SSSPContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "SSSP kernel time: " << traversal_kernel_time * 1000;
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    LOG(INFO) << "GPU " << dev << " Compute time: " << compute_time * 1000
              << " ms Comm time: " << mm->GetAccumulatedCommTime() * 1000
              << " ms Ratio: " << compute_time / mm->GetAccumulatedCommTime();
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id,
            int init_prio) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->src_id = src_id;
    this->lb = app_config.lb;
    dist.Init(vertices, std::numeric_limits<dist_t>::max());
    dist.H2D();

    auto in_cap = 2 * app_config.wl_alloc_factor_in * frag.GetEdgeNum();
    auto local_out_cap =
        (frag.fnum() == 1 ? app_config.wl_alloc_factor_in
                          : app_config.wl_alloc_factor_out_local) *
        frag.GetEdgeNum() * 2;
    auto remote_out_cap = ov.size();

    tmp_q.Init(iv.size());
    in_q.Init(iv);
    out_q_local_near.Init(iv);
    out_q_local_far.Init(iv);
    out_q_remote.Init(ov);

    double weight_sum = 0;

    for (auto v : iv) {
      auto oes = frag.GetOutgoingAdjList(v);
      for (auto& e : oes) {
        weight_sum += e.get_data();
      }
    }
    if (init_prio == 0) {
      /**
       * We select a similar heuristic, Δ = cw/d,
          where d is the average degree in the graph, w is the average
          edge weight, and c is the warp width (32 on our GPUs)
          Link: https://people.csail.mit.edu/jshun/papers/DBGO14.pdf
       */
      init_prio = 32 * (weight_sum / frag.GetEdgeNum()) /
                  (1.0 * frag.GetEdgeNum() / iv.size());
    }
    prio = init_prio;

#ifdef PROFILING
    VLOG(1) << "In size: " << in_cap << " Local out size: " << local_out_cap
            << " Remote out size: " << remote_out_cap;
#endif

    messages.InitBuffer((sizeof(vid_t) + sizeof(dist_t)) * remote_out_cap,
                        (sizeof(vid_t) + sizeof(dist_t)) * in_cap);
    mm = &messages;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    dist.D2H();

    for (auto v : iv) {
      if (dist[v] == std::numeric_limits<dist_t>::max()) {
        os << frag.GetId(v) << " infinity" << std::endl;
      } else {
        os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
           << dist[v] << std::endl;
      }
    }
  }

  oid_t src_id;
  LoadBalancing lb{};
  VertexArray<dist_t, vid_t> dist;
  Queue<vertex_t> tmp_q;
  DenseVertexSet<vid_t> in_q, out_q_local_near, out_q_local_far;
  DenseVertexSet<vid_t> out_q_remote;
  dist_t init_prio{};
  dist_t prio{};
#ifdef PROFILING
  double get_msg_time{};
  double traversal_kernel_time{};
  double compute_time{};
#endif
  GPUMessageManager* mm;
};

template <typename FRAG_T>
class SSSP : public GPUAppBase<FRAG_T, SSSPContext<FRAG_T>>,
             public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(SSSP<FRAG_T>, SSSPContext<FRAG_T>, FRAG_T)
  using dist_t = typename context_t::dist_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto src_id = ctx.src_id;
    vertex_t source;
    bool native_source = frag.GetInnerVertex(src_id, source);

    if (native_source) {
      LaunchKernel(
          messages.stream(),
          [=] __device__(dev_fragment_t d_frag,
                         dev::VertexArray<dist_t, vid_t> dist,
                         dev::DenseVertexSet<vid_t> in_q) {
            auto tid = TID_1D;

            if (tid == 0) {
              dist[source] = 0;
              in_q.Insert(source);
            }
          },
          frag.DeviceObject(), ctx.dist.DeviceObject(),
          ctx.in_q.DeviceObject());
    }
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_dist = ctx.dist.DeviceObject();
    auto& tmp_q = ctx.tmp_q;
    auto d_tmp_q = tmp_q.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local_near = ctx.out_q_local_near;
    auto& out_q_local_far = ctx.out_q_local_far;
    auto& out_q_remote = ctx.out_q_remote;
    auto d_out_q_remote = out_q_remote.DeviceObject();
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto d_mm = messages.DeviceObject();
    auto& prio = ctx.prio;
    auto iv = frag.InnerVertices();

#ifdef PROFILING
    ctx.get_msg_time -= grape::GetCurrentTime();
#endif
    messages.template ParallelProcess<dev_fragment_t, dist_t>(
        d_frag, [=] __device__(vertex_t v, dist_t received_dist) mutable {
          assert(d_frag.IsInnerVertex(v));

#ifdef INT_WEIGHT
          if (received_dist < atomicMin(&d_dist[v], received_dist)) {
#else
          if (received_dist < dev::atomicMinFloat(&d_dist[v], received_dist)) {
#endif
            d_in_q.Insert(v);
          }
        });

    size_t in_size = in_q.Count(stream);

#ifdef PROFILING
    ctx.get_msg_time += grape::GetCurrentTime();
    VLOG(1) << "Frag " << frag.fid() << " In: " << in_size;
#endif

    out_q_remote.Clear(stream);

    if (in_size > 0) {
#ifdef PROFILING
      auto traversal_kernel_time = grape::GetCurrentTime();
#endif
      auto d_in = in_q.DeviceObject();
      auto d_out_q_local_near = out_q_local_near.DeviceObject();
      auto d_out_q_local_far = out_q_local_far.DeviceObject();

      {
        WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());

        tmp_q.Clear(stream);
        ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
          if (d_in.Exist(v)) {
            d_tmp_q.AppendWarp(v);
          }
        });
      }

      WorkSourceArray<vertex_t> ws_in(tmp_q.data(), tmp_q.size(stream));

      stream.Sync();

#ifdef PROFILING
      ctx.compute_time -= grape::GetCurrentTime();
#endif

      ForEachOutgoingEdge(
          stream, d_frag, ws_in,
          [=] __device__(vertex_t u) { return d_dist[u]; },
          [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                         const nbr_t& nbr) mutable {
            dist_t new_depth = metadata.metadata + nbr.get_data();
            vertex_t v = nbr.get_neighbor();
#ifdef INT_WEIGHT
            if (new_depth < atomicMin(&d_dist[v], new_depth)) {
#else
            if (new_depth < dev::atomicMinFloat(&d_dist[v], new_depth)) {
#endif
              if (d_frag.IsInnerVertex(v)) {
                if (new_depth < prio) {
                  d_out_q_local_near.Insert(v);
                } else {
                  d_out_q_local_far.Insert(v);
                }
              } else {
                d_out_q_remote.Insert(v);
              }
            }
          },
          ctx.lb);

      stream.Sync();
#ifdef PROFILING
      ctx.compute_time += grape::GetCurrentTime();
#endif
      in_q.Clear(stream);

      auto local_size = out_q_local_near.Count(stream);

      if (local_size > 0) {
        in_q.Swap(out_q_local_near);
      } else {
        local_size = out_q_local_far.Count(stream);
        in_q.Swap(out_q_local_far);
        prio += ctx.init_prio;
      }

      if (local_size > 0) {
        messages.ForceContinue();
      }

#ifdef PROFILING
      traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
      ctx.traversal_kernel_time += traversal_kernel_time;
      VLOG(2) << "Frag " << frag.fid() << " Local out: " << local_size
              << " Kernel time: " << traversal_kernel_time * 1000;
#endif
    }

    {
      auto ov = frag.OuterVertices();
      auto ws_in = WorkSourceRange<vertex_t>(*ov.begin(), ov.size());

      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_out_q_remote.Exist(v)) {
          d_mm.template SyncStateOnOuterVertexWarpOpt(d_frag, v, d_dist[v]);
        }
      });
    }
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_SSSP_SSSP_H_
