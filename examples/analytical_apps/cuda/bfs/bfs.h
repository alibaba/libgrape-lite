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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_BFS_BFS_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_BFS_BFS_H_
#ifdef __CUDACC__
#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {

template <typename FRAG_T>
class BFSContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using depth_t = int64_t;

  explicit BFSContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~BFSContext() {
    LOG(INFO) << "Get msg time: " << get_msg_time * 1000;
    LOG(INFO) << "BFS kernel time: " << traversal_kernel_time * 1000;
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->src_id = src_id;
    this->lb = app_config.lb;
    depth.Init(vertices, std::numeric_limits<depth_t>::max());
    depth.H2D();
    in_q.Init(iv.size());
    out_q_local.Init(iv.size());

    messages.InitBuffer((sizeof(depth_t) + sizeof(vid_t)) * ov.size(),
                        (sizeof(depth_t) + sizeof(vid_t)) * iv.size());
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    depth.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << depth[v] << std::endl;
    }
  }

  oid_t src_id{};
  LoadBalancing lb{};
  depth_t curr_depth{};
  VertexArray<depth_t, vid_t> depth;
  Queue<vertex_t, vid_t> in_q, out_q_local;
#ifdef PROFILING
  double get_msg_time{};
  double traversal_kernel_time{};
#endif
};

template <typename FRAG_T>
class BFS : public GPUAppBase<FRAG_T, BFSContext<FRAG_T>>,
            public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(BFS<FRAG_T>, BFSContext<FRAG_T>, FRAG_T)
  using depth_t = typename context_t::depth_t;
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
                         dev::VertexArray<depth_t, vid_t> depth,
                         dev::Queue<vertex_t, vid_t> in_q) {
            auto tid = TID_1D;

            if (tid == 0) {
              depth[source] = 0;
              in_q.Append(source);
            }
          },
          frag.DeviceObject(), ctx.depth.DeviceObject(),
          ctx.in_q.DeviceObject());
    }
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_depth = ctx.depth.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local = ctx.out_q_local;
    auto d_out_q_local = out_q_local.DeviceObject();
    auto curr_depth = ctx.curr_depth;
    auto next_depth = curr_depth + 1;
    auto& stream = messages.stream();
    auto d_mm = messages.DeviceObject();

#ifdef PROFILING
    ctx.get_msg_time -= grape::GetCurrentTime();
    auto process_msg_time = grape::GetCurrentTime();
#endif
    messages.template ParallelProcess<dev_fragment_t, grape::EmptyType>(
        d_frag, [=] __device__(vertex_t v) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (curr_depth < d_depth[v]) {
            d_depth[v] = curr_depth;
            d_in_q.AppendWarp(v);
          }
        });
    auto in_size = in_q.size(stream);

    WorkSourceArray<vertex_t> ws_in(in_q.data(), in_size);

#ifdef PROFILING
    VLOG(1) << "Frag " << frag.fid() << " In: " << in_size;
    process_msg_time = grape::GetCurrentTime() - process_msg_time;
    ctx.get_msg_time += grape::GetCurrentTime();
    auto traversal_kernel_time = grape::GetCurrentTime();
#endif

    ForEachOutgoingEdge(
        stream, d_frag, ws_in,
        [=] __device__(const vertex_t& u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();

          if (next_depth < d_depth[v]) {
            d_depth[v] = next_depth;

            if (d_frag.IsInnerVertex(v)) {
              d_out_q_local.AppendWarp(v);
            } else {
              d_mm.SyncStateOnOuterVertex(d_frag, v);
            }
          }
        },
        ctx.lb);
    stream.Sync();
    auto local_out_size = out_q_local.size(stream);
#ifdef PROFILING
    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    VLOG(2) << "Frag " << frag.fid() << " Local out: " << local_out_size
            << " ProcessMsg time: " << process_msg_time * 1000
            << " Kernel time: " << traversal_kernel_time * 1000;
    ctx.traversal_kernel_time += traversal_kernel_time;
#endif
    in_q.Clear(stream);
    out_q_local.Swap(in_q);
    ctx.curr_depth = next_depth;
    if (local_out_size > 0) {
      messages.ForceContinue();
    }
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_BFS_BFS_H_
