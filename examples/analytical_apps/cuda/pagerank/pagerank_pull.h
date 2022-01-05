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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_PAGERANK_PAGERANK_PULL_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_PAGERANK_PAGERANK_PULL_H_
#ifdef __CUDACC__
#include "cuda/app_config.h"
#include "grape/cuda/utils/vertex_array.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class PagerankPullContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using rank_t = float;

  explicit PagerankPullContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  ~PagerankPullContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "Pagerank kernel time: " << traversal_kernel_time * 1000;
    VLOG(1) << "Send msg time: " << send_msg_time * 1000;
  }

  void Init(BatchShuffleMessageManager& messages, AppConfig app_config,
            float damping_factor, int max_iter) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    this->damping_factor = damping_factor;
    this->max_iter = max_iter;
    this->lb = app_config.lb;

    rank.Init(vertices, 0);
    rank.H2D();

    next_rank.Init(vertices, 0);
    next_rank.H2D();
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    rank.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << rank[v] << std::endl;
    }
  }

  int curr_iter{};
  int max_iter{};
  rank_t damping_factor{};
  rank_t dangling_sum{};
  LoadBalancing lb{};
  VertexArray<rank_t, vid_t> rank;
  VertexArray<rank_t, vid_t> next_rank;
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
};

template <typename FRAG_T>
class PagerankPull
    : public BatchShuffleAppBase<FRAG_T, PagerankPullContext<FRAG_T>>,
      public ParallelEngine,
      public Communicator {
 public:
  INSTALL_GPU_BATCH_SHUFFLE_WORKER(PagerankPull<FRAG_T>,
                                   PagerankPullContext<FRAG_T>, FRAG_T)
  using rank_t = typename context_t::rank_t;
  using device_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename device_t::vertex_t;
  using nbr_t = typename device_t::nbr_t;

  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kBothOutIn;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    ctx.curr_iter = 0;
    ctx.dangling_sum = 0;
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_next_rank = ctx.next_rank.DeviceObject();
    auto d_rank = ctx.rank.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto total_vertices_num = frag.GetTotalVerticesNum();
    auto damping_factor = ctx.damping_factor;
    auto& dangling_sum = ctx.dangling_sum;

    auto traversal_kernel_time = grape::GetCurrentTime();

    WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());

    rank_t local_dangling_sum = thrust::transform_reduce(
        thrust::device, thrust::make_counting_iterator(iv.begin().GetValue()),
        thrust::make_counting_iterator(iv.end().GetValue()),
        [=] __device__(vid_t lid) -> rank_t {
          vertex_t v(lid);

          return d_frag.GetLocalOutDegree(v) == 0 ? d_rank[v] : 0;
        },
        (rank_t) 0.0, thrust::plus<rank_t>());

    Sum(local_dangling_sum, dangling_sum);

    auto base = (1 - damping_factor) / total_vertices_num +
                damping_factor * dangling_sum / total_vertices_num;

    ForEach(stream, ws_in,
            [=] __device__(vertex_t v) mutable { d_next_rank[v] = base; });

    ForEachIncomingEdge(
        stream, d_frag, ws_in,
        [=] __device__(vertex_t v, const nbr_t& nbr) mutable {
          vertex_t u = nbr.get_neighbor();
          rank_t rank_to_pull = d_rank[u];

          atomicAdd(&d_next_rank[v], rank_to_pull);
        },
        ctx.lb);

    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    ctx.traversal_kernel_time += traversal_kernel_time;

    stream.Sync();
    VLOG(1) << "Frag " << frag.fid()
            << " Kernel time: " << traversal_kernel_time * 1000;

    if (ctx.curr_iter++ < ctx.max_iter - 1) {
      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_frag.GetLocalOutDegree(v) > 0) {
          d_next_rank[v] =
              damping_factor * d_next_rank[v] / d_frag.GetLocalOutDegree(v);
        }
      });

      messages.template SyncInnerVertices<fragment_t, rank_t>(frag,
                                                              ctx.next_rank);
    }

    ctx.rank.Swap(ctx.next_rank);
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_PAGERANK_PAGERANK_PULL_H_
