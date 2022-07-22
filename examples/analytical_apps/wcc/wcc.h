#ifndef EXAMPLES_ANALYTICAL_APPS_WCC_WCC_H_
#define EXAMPLES_ANALYTICAL_APPS_WCC_WCC_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class WCCContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using label_t = vid_t;

  explicit WCCContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

  ~WCCContext() {
  }
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

    in_q.Init(iv.size());
    out_q_local.Init(iv.size());
    out_q_remote.Init(ov.size());
    visited.Init(vertices);

    // <gid, gid> will be packed and sent to destinations
    messages.InitBuffer(
        ov.size() * (sizeof(vid_t) + sizeof(label_t)),
        iv.size() * (sizeof(vid_t) + sizeof(label_t)) * (frag.fnum() - 1));
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    label.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << frag.Gid2Oid(label[v]) << std::endl;
    }
  }

  LoadBalancing lb{};
  VertexArray<label_t, vid_t> label;
  Queue<vertex_t> in_q, out_q_local, out_q_remote;
  DenseVertexSet<vid_t> visited;
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

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto iv = frag.InnerVertices();
    WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());
    double compute = 0;
    auto& stream = messages.stream();

    compute -= grape::GetCurrentTime();
    LaunchKernel(
        stream,
        [=] __device__(dev_fragment_t d_frag,
                       dev::VertexArray<label_t, vid_t> label,
                       dev::Queue<vertex_t, uint32_t> d_in_q) mutable {
          auto tid = TID_1D;
          auto total_nthreads = TOTAL_THREADS_1D;
          auto size = ws_in.size();

          for (size_t idx = 0 + tid; idx < size; idx += total_nthreads) {
            vertex_t v = ws_in.GetWork(idx);

            label[v] = d_frag.Vertex2Gid(v);
            d_in_q.AppendWarp(v);
          }
        },
        frag.DeviceObject(), ctx.label.DeviceObject(), ctx.in_q.DeviceObject());
    stream.Sync();
    compute += grape::GetCurrentTime();

    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(compute);
    messages.RecordPackTime(0);
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
    auto& visited = ctx.visited;
    auto d_visited = visited.DeviceObject();
    auto d_frag = frag.DeviceObject();
    auto iv = frag.InnerVertices();
    auto& stream = messages.stream();
    auto d_mm = messages.DeviceObject();
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    t_unpack -= grape::GetCurrentTime();
    messages.template ParallelProcess<dev_fragment_t, label_t>(
        d_frag, [=] __device__(vertex_t v, label_t received_gid) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (received_gid < atomicMin(&d_label[v], received_gid)) {
            if(d_visited.Insert(v)) {
              d_in_q.AppendWarp(v);
            }
          }
        });
    stream.Sync();
    t_unpack += grape::GetCurrentTime();

    auto local_size = in_q.size(stream);

    visited.Clear(stream);
    out_q_local.Clear(stream);
    out_q_remote.Clear(stream);

    t_compute -= grape::GetCurrentTime();
    ForEachOutgoingEdge(
        stream, d_frag, WorkSourceArray<vertex_t>(in_q.data(), local_size),
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          label_t u_label = d_label[u];
          vertex_t v = nbr.get_neighbor();

          if (u_label < atomicMin(&d_label[v], u_label)) {
            if(d_visited.Insert(v)) {
              if (d_frag.IsInnerVertex(v)) {
                d_out_q_local.Append(v);
              } else {
                d_out_q_remote.Append(v);
              }
            }
          }
        },
        ctx.lb);
    stream.Sync();
    t_compute += grape::GetCurrentTime();


    t_pack -= grape::GetCurrentTime();
    messages.MakeOutput(stream, frag,
                        WorkSourceArray<vertex_t>(out_q_remote.data(),
                                                  out_q_remote.size(stream)),
                        [=] __device__(vertex_t v) mutable {
                          return thrust::make_pair(d_frag.GetOuterVertexGid(v),
                                                   d_label[v]);
                        });
    stream.Sync();
    t_pack += grape::GetCurrentTime();

    out_q_local.Swap(in_q);
    if (local_size > 0) {
      messages.ForceContinue();
    }

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }
};
}  // namespace grape_gpu
#endif  // EXAMPLES_ANALYTICAL_APPS_WCC_WCC_H_
