#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_FUSED_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_FUSED_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class BFSFusedContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using depth_t = uint32_t;

  explicit BFSFusedContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id) {
    auto& frag = this->fragment();
    auto nv = frag.GetTotalVerticesNum();
    auto iv = frag.InnerVertices();

    this->src_id = src_id;
    this->lb = app_config.lb;
    depth.resize(nv, std::numeric_limits<depth_t>::max());

    in_q.Init(nv);
    out_q.Init(nv);
    visited.Init(nv);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    thrust::host_vector<depth_t> h_depth = depth;

    if (native_source) {
      auto vm_ptr = frag.vm_ptr();

      for (vid_t sid = 0; sid < frag.GetTotalVerticesNum(); sid++) {
        auto gid = frag.ConsecutiveGid2Gid(sid);
        oid_t oid;

        CHECK(vm_ptr->GetOid(gid, oid));
        os << oid << " " << h_depth[sid] << "\n";
      }
    }
  }

  oid_t src_id;
  LoadBalancing lb{};
  thrust::device_vector<depth_t> depth;
  Queue<vid_t, uint32_t> in_q, out_q;
  Bitset<vid_t> visited;
  bool native_source;
};

template <typename FRAG_T>
class BFSFused : public GPUAppBase<FRAG_T, BFSFusedContext<FRAG_T>>,
                 public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(BFSFused<FRAG_T>, BFSFusedContext<FRAG_T>, FRAG_T)
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
    auto d_frag = frag.DeviceObject();

    ctx.native_source = frag.GetInnerVertex(src_id, source);

    if (ctx.native_source) {
      LaunchKernel(
          messages.stream(),
          [=] __device__(ArrayView<depth_t> depth,
                         dev::Queue<vid_t, uint32_t> in_q) {
            auto tid = TID_1D;

            if (tid == 0) {
              vid_t v_gid = d_frag.Vertex2Gid(source);
              vid_t v_sid = d_frag.Gid2Sid(v_gid);

              depth[v_sid] = 0;
              in_q.Append(v_gid);
            }
          },
          ArrayView<depth_t>(ctx.depth), ctx.in_q.DeviceObject());
    }
    messages.ForceContinue();

    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(0);
    messages.RecordPackTime(0);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_depth = ArrayView<depth_t>(ctx.depth);
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q = ctx.out_q;
    auto d_out_q = out_q.DeviceObject();
    auto& visited = ctx.visited;
    auto d_visited = visited.DeviceObject();
    auto& stream = messages.stream();
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    t_compute -= grape::GetCurrentTime();
    if (ctx.native_source) {
      auto d_local_frag = frag.DeviceObject();
      auto d_frags = frag.DeviceObjects();

      LaunchCooperativeKernel(stream, [=] __device__() mutable {
        auto grid = cooperative_groups::this_grid();
        size_t size;
        depth_t curr_depth = 0;

        while ((size = d_in_q.size()) > 0) {
          for (auto i = TID_1D; i < size; i += TOTAL_THREADS_1D) {
            vertex_t u;
            auto u_gid = d_in_q[i];
            auto fid = d_local_frag.GetFragId(u_gid);
            auto& d_frag = d_frags[fid];
            bool ok = d_frag.Gid2Vertex(u_gid, u);
            auto depth = d_depth[d_frag.Gid2Sid(u_gid)];
            assert(ok);

            for (auto e : d_frag.GetOutgoingAdjList(u)) {
              auto v = e.get_neighbor();
              auto v_gid = d_frag.Vertex2Gid(v);
              auto v_sid = d_frag.Gid2Sid(v_gid);
              auto new_depth = curr_depth + 1;

              if (new_depth < atomicMin(&d_depth[v_sid], new_depth)) {
                if (d_visited.set_bit_atomic(v_sid)) {
                  d_out_q.Append(v_gid);
                }
              }
            }
          }
          grid.sync();
          if (TID_1D == 0) {
            d_in_q.Clear();
          }
          d_visited.clear();
          d_in_q.Swap(d_out_q);
          curr_depth += 1;
          grid.sync();
        }
      });
      stream.Sync();
    }
    t_compute += grape::GetCurrentTime();

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }
};
}  // namespace grape_gpu

#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_FUSED_H_
