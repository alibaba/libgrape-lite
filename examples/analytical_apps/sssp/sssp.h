#ifndef EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_H_
#define EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class SSSPContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using dist_t = uint32_t;
  explicit SSSPContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->src_id = src_id;
    this->lb = app_config.lb;
    dist.Init(vertices, std::numeric_limits<dist_t>::max());
    dist.H2D();

    in_q.Init(iv.size());
    out_q_local.Init(iv.size());
    out_q_remote.Init(ov.size());
    visited.Init(vertices);

    messages.InitBuffer(
        (sizeof(vid_t) + sizeof(dist_t)) * ov.size(),
        (sizeof(vid_t) + sizeof(dist_t)) * iv.size() * (frag.fnum() - 1));
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    dist.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << dist[v] << std::endl;
    }
  }

  oid_t src_id;
  LoadBalancing lb{};
  VertexArray<dist_t, vid_t> dist;
  Queue<vertex_t, uint32_t> in_q, out_q_local, out_q_remote;
  DenseVertexSet<vid_t> visited;
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
                         dev::Queue<vertex_t, uint32_t> in_q) {
            auto tid = TID_1D;

            if (tid == 0) {
              dist[source] = 0;
              in_q.Append(source);
            }
          },
          frag.DeviceObject(), ctx.dist.DeviceObject(),
          ctx.in_q.DeviceObject());
    }
    messages.ForceContinue();

    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(0);
    messages.RecordPackTime(0);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_mm = messages.DeviceObject();
    auto d_dist = ctx.dist.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local = ctx.out_q_local;
    auto d_out_q_local = out_q_local.DeviceObject();
    auto& out_q_remote = ctx.out_q_remote;
    auto d_out_q_remote = out_q_remote.DeviceObject();
    auto& visited = ctx.visited;
    auto d_visited = visited.DeviceObject();
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    t_unpack -= grape::GetCurrentTime();
    messages.template ParallelProcess<dev_fragment_t, dist_t>(
        d_frag, [=] __device__(vertex_t v, dist_t received_dist) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (received_dist < atomicMin(&d_dist[v], received_dist)) {
            if (d_visited.Insert(v)) {
              d_in_q.AppendWarp(v);
            }
          }
        });
    stream.Sync();
    t_unpack += grape::GetCurrentTime();

    t_compute -= grape::GetCurrentTime();
    size_t in_size = in_q.size(stream);

    out_q_local.Clear(stream);
    out_q_remote.Clear(stream);
    visited.Clear(stream);

    ForEachOutgoingEdge(
        stream, d_frag, WorkSourceArray<vertex_t>(in_q.data(), in_size),
        [=] __device__(vertex_t u) { return d_dist[u]; },
        [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                       const nbr_t& nbr) mutable {
          dist_t new_depth = metadata.metadata + nbr.get_data();
          vertex_t v = nbr.get_neighbor();
          if (new_depth < atomicMin(&d_dist[v], new_depth)) {
            if (d_visited.Insert(v)) {
              if (d_frag.IsInnerVertex(v)) {
                d_out_q_local.Append(v);
              } else {
                d_out_q_remote.Append(v);
              }
            }
          }
        },
        ctx.lb);
    auto local_size = out_q_local.size(stream);

    VLOG(2) << "fid: " << frag.fid() << " In: " << in_size
            << " Local out: " << local_size
            << " Remote out: " << out_q_remote.size(stream);

    in_q.Swap(out_q_local);

    if (local_size > 0) {
      messages.ForceContinue();
    }
    stream.Sync();
    t_compute += grape::GetCurrentTime();

    t_pack -= grape::GetCurrentTime();
    messages.MakeOutput(stream, frag,
                        WorkSourceArray<vertex_t>(out_q_remote.data(),
                                                  out_q_remote.size(stream)),
                        [=] __device__(vertex_t v) mutable {
                          return thrust::make_pair(d_frag.GetOuterVertexGid(v),
                                                   d_dist[v]);
                        });
    stream.Sync();
    t_pack += grape::GetCurrentTime();

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }
};
}  // namespace grape_gpu

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_H_
