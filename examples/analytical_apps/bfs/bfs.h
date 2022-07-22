#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_H_
#include "VariadicTable.h"
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"
#include "grape_gpu/utils/time_table.h"

namespace grape_gpu {
template <typename FRAG_T>
class BFSContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using depth_t = uint32_t;

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
    out_q_local.Init(iv.size());
    out_q_remote.Init(ov.size());

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
  Queue<vertex_t, vid_t> in_q, out_q_local, out_q_remote;
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


    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(0);
    messages.RecordPackTime(0);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_depth = ctx.depth.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local = ctx.out_q_local;
    auto d_out_q_local = out_q_local.DeviceObject();
    auto& out_q_remote = ctx.out_q_remote;
    auto d_out_q_remote = out_q_remote.DeviceObject();
    auto curr_depth = ctx.curr_depth;
    auto next_depth = curr_depth + 1;
    auto& stream = messages.stream();
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    t_unpack -= grape::GetCurrentTime();
    messages.template ParallelProcess<dev_fragment_t, grape::EmptyType>(
        d_frag, [=] __device__(vertex_t v) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (curr_depth < d_depth[v]) {
            d_depth[v] = curr_depth;
            d_in_q.AppendWarp(v);
          }
        });
    stream.Sync();
    t_unpack += grape::GetCurrentTime();

    auto in_size = in_q.size(stream);
    WorkSourceArray<vertex_t> ws_in(in_q.data(), in_size);

    out_q_local.Clear(stream);
    out_q_remote.Clear(stream);

    t_compute -= grape::GetCurrentTime();
    ForEachOutgoingEdge(
        stream, d_frag, ws_in,
        [=] __device__(const vertex_t& u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();

          if (next_depth < d_depth[v]) {
            d_depth[v] = next_depth;
            if (d_frag.IsInnerVertex(v)) {
              d_out_q_local.Append(v);
            } else {
              d_out_q_remote.Append(v);
            }
          }
        },
        ctx.lb);
    stream.Sync();
    t_compute += grape::GetCurrentTime();

    auto local_out_size = out_q_local.size(stream);
    VLOG(1) << "Frag " << frag.fid() << " In: " << in_size
            << " Out: " << local_out_size;

    t_pack -= grape::GetCurrentTime();
    messages.MakeOutput(stream, frag,
                        WorkSourceArray<vertex_t>(out_q_remote.data(),
                                                  out_q_remote.size(stream)),
                        [=] __device__(vertex_t v) mutable {
                          return d_frag.GetOuterVertexGid(v);
                        });
    stream.Sync();
    t_pack += grape::GetCurrentTime();

    out_q_local.Swap(in_q);
    ctx.curr_depth = next_depth;
    if (local_out_size > 0) {
      messages.ForceContinue();
    }

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }
};
}  // namespace grape_gpu
#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_H_
