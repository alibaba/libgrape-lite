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
    current_active_map.Init(iv);
    next_active_map.Init(iv);
    visited.Init(iv);

    messages.InitBuffer((sizeof(vid_t)) * ov.size(),
                        (sizeof(vid_t)) * iv.size());
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
  int depth_to_use_pull;
  LoadBalancing lb{};
  depth_t curr_depth{};
  VertexArray<depth_t, vid_t> depth;
  Queue<vertex_t, vid_t> in_q;
  DenseVertexSet<vid_t> current_active_map, next_active_map;
  DenseVertexSet<vid_t> visited;
};

template <typename FRAG_T>
class BFS : public GPUAppBase<FRAG_T, BFSContext<FRAG_T>>,
            public ParallelEngine,
            public Communicator {
 public:
  INSTALL_GPU_WORKER(BFS<FRAG_T>, BFSContext<FRAG_T>, FRAG_T)
  using depth_t = typename context_t::depth_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;
  static constexpr bool need_split_edges = true;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto src_id = ctx.src_id;
    vertex_t source;
    bool native_source = frag.GetInnerVertex(src_id, source);
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;

    if (native_source) {
      LaunchKernel(
          messages.stream(),
          [=] __device__(dev_fragment_t d_frag,
                         dev::VertexArray<depth_t, vid_t> depth,
                         dev::DenseVertexSet<vid_t> d_current_active_map,
                         dev::DenseVertexSet<vid_t> d_visited) {
            auto tid = TID_1D;

            if (tid == 0) {
              depth[source] = 0;
              d_current_active_map.Insert(source);
              d_visited.Insert(source);
            }
          },
          frag.DeviceObject(), ctx.depth.DeviceObject(),
          ctx.current_active_map.DeviceObject(), ctx.visited.DeviceObject());
    }
    size_t edge_num = frag.GetOutgoingEdgeNum() +
                      (isDirected ? frag.GetIncomingEdgeNum() : 0);
    size_t total_edge_num;
    Sum(edge_num, total_edge_num);
    size_t total_vertex_num = frag.GetTotalVerticesNum();
    double avg_degree = (total_edge_num / 2.0) / total_vertex_num;
    if (avg_degree > 45) {
      ctx.depth_to_use_pull = -1;
    } else if (avg_degree > 20 && avg_degree < 40) {
      if (total_vertex_num > 50000000 && avg_degree < 30) {
        ctx.depth_to_use_pull = 4;
      } else {
        ctx.depth_to_use_pull = 2;
      }
    } else {
      ctx.depth_to_use_pull = -1;
    }
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto d_depth = ctx.depth.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& current_active_map = ctx.current_active_map;
    auto d_current_active_map = current_active_map.DeviceObject();
    auto& visited = ctx.visited;
    auto d_visited = visited.DeviceObject();
    auto& next_active_map = ctx.next_active_map;
    auto d_next_active_map = next_active_map.DeviceObject();
    auto curr_depth = ctx.curr_depth;
    auto next_depth = curr_depth + 1;
    auto& stream = messages.stream();
    auto d_mm = messages.DeviceObject();
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;

    next_active_map.Clear(stream);
    in_q.Clear(stream);

    messages.template ParallelProcess<dev_fragment_t, grape::EmptyType>(
        d_frag, [=] __device__(vertex_t v) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (curr_depth < d_depth[v]) {
            d_depth[v] = curr_depth;
            d_current_active_map.Insert(v);
            d_visited.Insert(v);
          }
        });

    auto ivnum = iv.size();
    auto active = current_active_map.Count(stream);
    auto visited_num = visited.Count(stream);
    double active_ratio = (active + 0.0) / ivnum;
    double visited_ratio = (visited_num + 0.0) / ivnum;
    bool usePush = (2.5 * active_ratio < (1 - visited_ratio));
    if (ctx.depth_to_use_pull >= 0) {
      if (ctx.depth_to_use_pull > ctx.curr_depth) {
        usePush = true;
      } else {
        usePush = false;
      }
    }
    if (active == 0 || usePush) {
      // push-based search
      WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());
      ForEach(stream, ws_iv, [=] __device__(vertex_t v) mutable {
        if (d_current_active_map.Exist(v)) {
          d_in_q.AppendWarp(v);
        }
      });
      WorkSourceArray<vertex_t> ws_in(in_q.data(), in_q.size(stream));

      ForEachOutgoingEdge(
          stream, d_frag, ws_in,
          [=] __device__(const vertex_t& u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            if (next_depth < d_depth[v]) {
              d_depth[v] = next_depth;
              if (d_frag.IsInnerVertex(v)) {
                d_next_active_map.Insert(v);
                d_visited.Insert(v);
              } else {
                d_mm.SyncStateOnOuterVertexWarpOpt(d_frag, v);
              }
            }
          },
          ctx.lb);
    } else {
      // pull-based search
      WorkSourceRange<vertex_t> ws_ov(*ov.begin(), ov.size());
      depth_t MAX_DEPTH = std::numeric_limits<depth_t>::max();
      ForEach(stream, ws_ov, [=] __device__(vertex_t v) mutable {
        if (d_depth[v] == MAX_DEPTH) {
          auto ies = d_frag.GetIncomingAdjList(v);
          for (auto& e : ies) {
            auto u = e.get_neighbor();
            assert(d_frag.IsInnerVertex(u));
            if (d_current_active_map.Exist(u)) {
              d_depth[v] = next_depth;
              d_mm.SyncStateOnOuterVertexWarpOpt(d_frag, v);
              break;
            }
          }
        }
      });

      WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());
      ForEach(stream, ws_iv, [=] __device__(vertex_t v) mutable {
        if (!d_visited.Exist(v)) {
          d_in_q.AppendWarp(v);
        }
      });
      WorkSourceArray<vertex_t> ws_in(in_q.data(), in_q.size(stream));

      if (isDirected) {
        ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
          auto ies = d_frag.GetIncomingInnerVertexAdjList(v);
          for (auto& e : ies) {
            auto u = e.get_neighbor();
            if (d_current_active_map.Exist(u)) {
              d_depth[v] = next_depth;
              d_next_active_map.Insert(v);
              d_visited.Insert(v);
              break;
            }
          }
        });
      } else {
        ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
          auto oes = d_frag.GetOutgoingInnerVertexAdjList(v);
          for (auto& e : oes) {
            auto u = e.get_neighbor();
            assert(d_frag.IsInnerVertex(u));
            if (d_current_active_map.Exist(u)) {
              d_depth[v] = next_depth;
              d_next_active_map.Insert(v);
              d_visited.Insert(v);
              break;
            }
          }
        });
      }
    }

    auto has_work = next_active_map.Count(stream);
    stream.Sync();

    if (has_work > 0) {
      messages.ForceContinue();
    }

    ctx.curr_depth = next_depth;
    current_active_map.Swap(next_active_map);
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_BFS_BFS_H_
