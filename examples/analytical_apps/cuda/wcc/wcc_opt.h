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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_WCC_WCC_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_WCC_WCC_OPT_H_
#ifdef __CUDACC__
#include "cuda/app_config.h"
#include "grape/fragment/id_parser.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class WCCOptContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using offset_t = vid_t;

  explicit WCCOptContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config) {
    auto& frag = this->fragment();

    // After being converted to coo, the internal csr/csc is released
    coo_frag = const_cast<FRAG_T&>(frag).ConvertToCOO(true);
    parents.resize(frag.GetTotalVerticesNum());
    // Init parent id
    LaunchKernel(
        messages.stream(),
        [=] __device__(ArrayView<offset_t> parents) {
          auto tid = TID_1D;
          auto nthreads = TOTAL_THREADS_1D;

          for (size_t i = 0 + tid; i < parents.size(); i += nthreads) {
            parents[i] = i;
          }
        },
        ArrayView<offset_t>(parents));

    this->lb = app_config.lb;
    id_parser.init(frag.fnum());

    if (frag.fid() == 0) {
      messages.InitBuffer(
          0, (frag.GetTotalVerticesNum() - frag.InnerVertices().size()) *
                 sizeof(offset_t) * 2 * 2);
    } else {
      messages.InitBuffer(frag.Vertices().size() * sizeof(offset_t) * 2 * 1.5,
                          0);
    }
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();

    if (frag.fid() == 0) {
      size_t graph_vnum = frag.GetTotalVerticesNum();
      pinned_vector<offset_t> tmp = v_offset;

      auto offset2gid = [&tmp, this](offset_t offset) {
        fid_t fid = std::lower_bound(tmp.begin(), tmp.end(), offset + 1) -
                    tmp.begin() - 1;
        offset_t begin_offset = tmp[fid];
        offset_t lid = offset - begin_offset;

        return id_parser.generate_global_id(fid, lid);
      };

      thrust::host_vector<offset_t> h_parents = parents;

      for (offset_t offset = 0; offset < graph_vnum; offset++) {
        auto gid = offset2gid(offset);
        auto p_gid = offset2gid(h_parents[offset]);
        os << frag.Gid2Oid(gid) << " " << p_gid << std::endl;
      }
    }
  }

  LoadBalancing lb{};
  IdParser<vid_t> id_parser;
  thrust::device_vector<offset_t> parents;
  thrust::device_vector<offset_t> v_offset;
  std::shared_ptr<typename FRAG_T::coo_t> coo_frag;
  double get_msg_time{};
  double traversal_kernel_time{};
};

template <typename FRAG_T>
class WCCOpt : public GPUAppBase<FRAG_T, WCCOptContext<FRAG_T>>,
               public Communicator,
               public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(WCCOpt<FRAG_T>, WCCOptContext<FRAG_T>, FRAG_T)
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;
  using offset_t = typename context_t::offset_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_coo = ctx.coo_frag->DeviceObject();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto av = frag.Vertices();
    auto dev_mm = messages.DeviceObject();
    auto* d_parents = thrust::raw_pointer_cast(ctx.parents.data());
    auto& stream = messages.stream();
    auto id_parser = ctx.id_parser;

    // Calculate offset array
    {
      pinned_vector<offset_t> tmp;
      offset_t offset = 0;

      for (auto nv : AllGather(iv.size())) {
        tmp.push_back(offset);
        offset += nv;
      }
      ctx.v_offset = tmp;
    }

    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
    auto* d_v_offset = thrust::raw_pointer_cast(ctx.v_offset.data());

    auto vertex2offset = [=] __device__(vertex_t v) -> offset_t {
      auto gid = d_frag.Vertex2Gid(v);
      auto fid = id_parser.get_fragment_id(gid);
      auto lid = id_parser.get_local_id(gid);

      return d_v_offset[fid] + lid;
    };

    // HookHighToLowAtomic
    size_t begin_eid = 0;
    size_t batch_size = ws_in.size();

    while (begin_eid < d_coo.GetEdgeNum()) {
      size_t end_eid = std::min(begin_eid + batch_size, d_coo.GetEdgeNum());

      LaunchKernel(stream, batch_size, [=] __device__() mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (size_t i = begin_eid + tid; i < end_eid; i += nthreads) {
          auto& e = d_coo[i];
          auto u = vertex_t(e.src);
          auto v = vertex_t(e.dst);

          // is not a self-cycle
          if (u != v) {
            auto u_offset = vertex2offset(u);
            auto v_offset = vertex2offset(v);
            auto p_u = d_parents[u_offset];
            auto p_v = d_parents[v_offset];

            while (p_u != p_v) {
              auto high = p_u > p_v ? p_u : p_v;
              auto low = p_u + p_v - high;
              auto prev = atomicCAS(&d_parents[high], high, low);

              if (prev == high || prev == low) {
                break;
              }
              p_u = d_parents[prev];
              p_v = d_parents[low];
            }
          }
        }
      });

      // MultiJumpCompress
      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        auto v_offset = vertex2offset(v);
        auto p = d_parents[v_offset];
        auto pp = d_parents[p];

        while (p != pp) {
          d_parents[v_offset] = pp;
          p = pp;
          pp = d_parents[p];
        }
      });
      begin_eid = end_eid;
    }

    if (frag.fid() > 0) {
      WorkSourceRange<vertex_t> ws_in(*av.begin(), av.size());

      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        auto offset = vertex2offset(v);
        auto parent = d_parents[offset];

        if (offset != parent) {
          dev_mm.template SendToFragmentWarpOpt(
              0, thrust::make_pair(offset, parent));
        }
      });
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto* d_parents = thrust::raw_pointer_cast(ctx.parents.data());
    auto* d_v_offset = thrust::raw_pointer_cast(ctx.v_offset.data());
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto id_parser = ctx.id_parser;

    auto vertex2offset = [=] __device__(vertex_t v) -> offset_t {
      auto gid = d_frag.Vertex2Gid(v);
      auto fid = id_parser.get_fragment_id(gid);
      auto lid = id_parser.get_local_id(gid);

      return d_v_offset[fid] + lid;
    };

    auto hook_high_to_low = [=] __device__(offset_t offset, offset_t parent) {
      if (offset != parent) {
        auto p_u = d_parents[offset];
        auto p_v = d_parents[parent];

        while (p_u != p_v) {
          auto high = p_u > p_v ? p_u : p_v;
          auto low = p_u + p_v - high;
          auto prev = atomicCAS(&d_parents[high], high, low);

          if (prev == high || prev == low) {
            break;
          }

          p_u = d_parents[prev];
          p_v = d_parents[low];
        }
      }
    };  // NOLINT

    // HookHighToLowAtomic
    messages.template ParallelProcess<thrust::pair<offset_t, offset_t>>(
        [=] __device__(const thrust::pair<offset_t, offset_t>& msg) mutable {
          auto offset = msg.first;
          auto parent = msg.second;

          hook_high_to_low(offset, parent);
        });

    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());

    ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
      auto offset = vertex2offset(v);
      auto parent = d_parents[offset];

      hook_high_to_low(offset, parent);
    });

    if (frag.fnum() > 1 && frag.fid() == 0) {
      size_t graph_vnum = frag.GetTotalVerticesNum();
      // MultiJumpCompress
      LaunchKernel(stream, [=] __device__() mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (int v_offset = 0 + tid; v_offset < graph_vnum;
             v_offset += nthreads) {
          auto p = d_parents[v_offset];
          auto pp = d_parents[p];

          while (p != pp) {
            d_parents[v_offset] = pp;
            p = pp;
            pp = d_parents[p];
          }
        }
      });
    }
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_WCC_WCC_OPT_H_
