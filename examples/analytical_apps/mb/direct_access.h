
#ifndef GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_MB_DIRECT_ACCESS_H_
#define GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_MB_DIRECT_ACCESS_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class MBDirectAccessContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using dist_t = uint32_t;

  explicit MBDirectAccessContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config,
            float steal_factor) {
    auto& frag = this->fragment();
    auto vertices = frag.LocalVertices(0);
    auto iv = frag.LocalInnerVertices(0);
    auto ov = frag.OuterVertices(0);

    this->src_id = src_id;
    this->lb = app_config.lb;
    this->steal_factor = steal_factor;
    dist.Init(messages.comm_spec(), vertices,
              std::numeric_limits<dist_t>::max());
    dist.H2D();
    out_q.Init(messages.comm_spec(), iv.size());
    visited.Init(messages.comm_spec(), vertices);
    CHECK_EQ(frag.fnum(), 2);

    messages.InitBuffer(
        (sizeof(vid_t) + sizeof(dist_t)) * ov.size(),
        (sizeof(vid_t) + sizeof(dist_t)) * iv.size() * (frag.fnum() - 1));
  }

  void Output(std::ostream& os) override {}

  oid_t src_id;
  LoadBalancing lb{};
  RemoteVertexArray<dist_t, vid_t> dist;
  RemoteQueue<vertex_t> out_q;
  RemoteDenseVertexSet<vid_t> visited;
  float steal_factor;
};

template <typename FRAG_T>
class MBDirectAccess : public GPUAppBase<FRAG_T, MBDirectAccessContext<FRAG_T>>,
                       public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(MBDirectAccess<FRAG_T>, MBDirectAccessContext<FRAG_T>,
                     FRAG_T)
  using dist_t = typename context_t::dist_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto d_frag = frag.DeviceObject(0);
    auto d_visited = ctx.visited.DeviceObject(0);
    auto& stream = messages.stream();
    double t_unpack = 0, t_compute = 0, t_pack = 0;
    auto iv = frag.LocalInnerVertices(0);
    vid_t v_size = iv.size();
    auto remote_size = v_size * ctx.steal_factor;
    auto local_size = v_size - remote_size;
    WorkSourceRange<vertex_t> ws;

    if (frag.fid() == 0) {
      ws = WorkSourceRange<vertex_t>(iv.begin(), local_size);
    } else {
      ws = WorkSourceRange<vertex_t>(
          vertex_t(iv.begin().GetValue() + local_size), remote_size);
    }

    LOG(INFO) << "fid: " << frag.fid() << " ws: " << ws.size();

    auto d_dist = ctx.dist.DeviceObject(0);
    auto d_out_q = ctx.out_q.DeviceObject(0);

    ForEach(stream, ws, [=] __device__(vertex_t v) mutable { d_dist[v] = 1; });
    stream.Sync();

    MPI_Barrier(messages.comm_spec().local_comm());

    t_compute -= grape::GetCurrentTime();
    ForEachOutgoingEdge(
        stream, d_frag, ws, [=] __device__(vertex_t u) { return d_dist[u]; },
        [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                       const nbr_t& nbr) mutable {
          dist_t new_dist = metadata.metadata + nbr.get_data();
          vertex_t v = nbr.get_neighbor();

          if (new_dist < atomicMin(&d_dist[v], new_dist)) {
            if (d_visited.Insert(v)) {
              d_out_q.AppendWarp(v);
            }
          }
        },
        ctx.lb);
    stream.Sync();
    t_compute += grape::GetCurrentTime();

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {}
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_MB_DIRECT_ACCESS_H_
