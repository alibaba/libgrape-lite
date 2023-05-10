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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_PAGERANK_PAGERANK_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_PAGERANK_PAGERANK_H_
#ifdef __CUDACC__
#include "cuda/app_config.h"
#include "grape/grape.h"
namespace grape {
namespace cuda {

template <grape::LoadStrategy LS>
struct RankTrait {
  using rank_t = float;
};

template <>
struct RankTrait<grape::LoadStrategy::kBothOutIn> {
  using rank_t = double;
};

template <typename FRAG_T>
class PagerankContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using rank_t = typename RankTrait<FRAG_T::load_strategy>::rank_t;

  explicit PagerankContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~PagerankContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "Pagerank kernel time: " << traversal_kernel_time * 1000;
    VLOG(1) << "Send msg time: " << send_msg_time * 1000;
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    LOG(INFO) << "GPU " << dev << " Compute time: " << compute_time * 1000
              << " ms Comm time: " << mm->GetAccumulatedCommTime() * 1000
              << " ms Ratio: " << compute_time / mm->GetAccumulatedCommTime();
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config,
            rank_t damping_factor, int max_iter) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    this->damping_factor = damping_factor;
    this->max_iter = max_iter;
    this->lb = app_config.lb;

    rank.Init(vertices, 0);
    rank.H2D();
    scale_factor = 1.0;

    next_rank.Init(vertices, 0);
    next_rank.H2D();

    auto comm_vol_in_bytes =
        frag.OuterVertices().size() * sizeof(thrust::pair<vid_t, rank_t>);

    messages.InitBuffer(comm_vol_in_bytes, comm_vol_in_bytes);
    mm = &messages;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    rank.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << rank[v] / (scale_factor) << std::endl;
    }
  }

  int curr_iter{};
  int max_iter{};
  rank_t damping_factor{};
  rank_t dangling_sum{};
  rank_t scale_factor;
  LoadBalancing lb{};
  VertexArray<rank_t, vid_t> rank;
  VertexArray<rank_t, vid_t> next_rank;
#ifdef PROFILING
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
  double compute_time{};
  double comm_time{};
#endif
  GPUMessageManager* mm;
};

template <typename FRAG_T>
class Pagerank : public GPUAppBase<FRAG_T, PagerankContext<FRAG_T>>,
                 public ParallelEngine,
                 public Communicator {
 public:
  INSTALL_GPU_WORKER(Pagerank<FRAG_T>, PagerankContext<FRAG_T>, FRAG_T)
  using rank_t = typename context_t::rank_t;
  using device_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename device_t::vertex_t;
  using nbr_t = typename device_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    ctx.curr_iter = 0;
    ctx.dangling_sum = 0;
    auto& stream = messages.stream();
    auto d_rank = ctx.rank.DeviceObject();
    auto iv = frag.InnerVertices();

    rank_t p = 1.0 * ctx.scale_factor / frag.GetTotalVerticesNum();

    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
    ForEach(stream, ws_in,
            [=] __device__(vertex_t v) mutable { d_rank[v] = p; });
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_next_rank = ctx.next_rank.DeviceObject();
    auto d_rank = ctx.rank.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto total_vertices_num = frag.GetTotalVerticesNum();
    auto damping_factor = ctx.damping_factor;
    auto& dangling_sum = ctx.dangling_sum;

#ifdef PROFILING
    ctx.get_msg_time -= grape::GetCurrentTime();
#endif
    messages.template ParallelProcess<device_t, rank_t>(
        d_frag, [=] __device__(vertex_t v, rank_t msg) mutable {
          assert(d_frag.IsInnerVertex(v));
          atomicAdd(&d_rank[v], msg);
        });
#ifdef PROFILING
    ctx.get_msg_time += grape::GetCurrentTime();
#endif

#ifdef PROFILING
    auto traversal_kernel_time = grape::GetCurrentTime();
#endif

    if (ctx.curr_iter++ >= ctx.max_iter) {
      return;
    } else {
      messages.ForceContinue();
    }

#ifdef PROFILING
    double begin = grape::GetCurrentTime();
#endif

    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());

    rank_t local_dangling_sum = thrust::transform_reduce(
        thrust::device, thrust::make_counting_iterator(iv.begin_value()),
        thrust::make_counting_iterator(iv.end_value()),
        [=] __device__(vid_t lid) -> rank_t {
          vertex_t v(lid);

          return d_frag.GetLocalOutDegree(v) == 0 ? d_rank[v] : 0;
        },
        (rank_t) 0.0, thrust::plus<rank_t>());

    Sum(local_dangling_sum, dangling_sum);

    rank_t scale_factor = ctx.scale_factor;
    ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
      d_next_rank[v] =
          scale_factor * (1 - damping_factor) / total_vertices_num +
          damping_factor * dangling_sum / total_vertices_num;
    });

    stream.Sync();

#ifdef PROFILING
    ctx.compute_time -= grape::GetCurrentTime();
#endif
    ForEachOutgoingEdge(
        stream, d_frag, ws_in,
        [=] __device__ __host__(vertex_t u) -> rank_t {
          rank_t rank_send =
              damping_factor * d_rank[u] / d_frag.GetLocalOutDegree(u);
          return rank_send;
        },
        [=] __device__(const VertexMetadata<vid_t, rank_t>& u_and_rank,
                       const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          rank_t rank_to_send = u_and_rank.metadata;

          atomicAdd(&d_next_rank[v], rank_to_send);
        },
        ctx.lb);
    stream.Sync();
#ifdef PROFILING
    ctx.compute_time += grape::GetCurrentTime();
#endif

    for (fid_t fid = 0; fid < frag.fnum(); fid++) {
      ov = frag.OuterVertices(fid);
      ws_in = WorkSourceRange<vertex_t>(*ov.begin(), ov.size());

      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_next_rank[v] > 0) {
          d_mm.template SyncStateOnOuterVertexWarpOpt(d_frag, v,
                                                      d_next_rank[v]);
          d_next_rank[v] = 0;
        }
      });
    }

#ifdef PROFILING
    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    ctx.traversal_kernel_time += traversal_kernel_time;
#endif

#ifdef PROFILING
    VLOG(1) << "Frag " << frag.fid()
            << " Kernel time: " << traversal_kernel_time * 1000;
#endif

    ctx.rank.Swap(ctx.next_rank);
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_PAGERANK_PAGERANK_H_
