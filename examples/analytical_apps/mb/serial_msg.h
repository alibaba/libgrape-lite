
#ifndef GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_MB_SERIAL_MSG_H_
#define GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_MB_SERIAL_MSG_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class MBSerialMsgContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using dist_t = uint32_t;

  explicit MBSerialMsgContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config,
            float steal_factor) {
    auto& frag = this->fragment();
    auto vertices = frag.LocalVertices(0);

    this->src_id = src_id;
    this->lb = app_config.lb;
    this->steal_factor = steal_factor;
    dist.Init(vertices, std::numeric_limits<dist_t>::max());
    dist.H2D();
    visited.Init(vertices.size());
    out_q.Init(vertices.size());
    CHECK_EQ(frag.fnum(), 2);

    messages.InitBuffer((sizeof(vid_t) + sizeof(dist_t)) * vertices.size(),
                        (sizeof(vid_t) + sizeof(dist_t)) * vertices.size());
  }

  void Output(std::ostream& os) override {}

  oid_t src_id;
  LoadBalancing lb{};
  VertexArray<dist_t, vid_t> dist;
  DenseVertexSet<vid_t> visited;
  Queue<vertex_t> out_q;
  float steal_factor;
};

template <typename FRAG_T>
class MBSerialMsg : public GPUAppBase<FRAG_T, MBSerialMsgContext<FRAG_T>>,
                    public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(MBSerialMsg<FRAG_T>, MBSerialMsgContext<FRAG_T>, FRAG_T)
  using dist_t = typename context_t::dist_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto d_frag = frag.DeviceObject(0);
    auto d_mm = messages.DeviceObject();
    auto d_visited = ctx.visited.DeviceObject();
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

    auto d_dist = ctx.dist.DeviceObject();
    auto d_out_q = ctx.out_q.DeviceObject();

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

    t_pack -= grape::GetCurrentTime();
    if (frag.fid() == 1) {
      ForEach(
          stream,
          WorkSourceArray<vertex_t>(ctx.out_q.data(), ctx.out_q.size(stream)),
          [=] __device__(vertex_t v) mutable {
            auto fid = d_frag.GetFragId(v);
            if (fid == 0) {
              d_mm.template SendToFragment(
                  fid, thrust::make_pair(d_frag.Vertex2Gid(v), d_dist[v]));
            }
          });
      stream.Sync();
    }
    t_pack += grape::GetCurrentTime();

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto& stream = messages.stream();
    auto d_frag = frag.DeviceObject(0);
    auto d_dist = ctx.dist.DeviceObject();
    auto d_visited = ctx.visited.DeviceObject();
    auto d_out_q = ctx.out_q.DeviceObject();
    double t_unpack = 0;

    if (frag.fid() == 0) {
      t_unpack -= grape::GetCurrentTime();
      messages.template ParallelProcess<dev_fragment_t, dist_t>(
          d_frag, [=] __device__(vertex_t v, dist_t received_dist) mutable {
            assert(d_frag.IsInnerVertex(v));
            if (received_dist < atomicMin(&d_dist[v], received_dist)) {
              if (d_visited.Insert(v)) {
                d_out_q.AppendWarp(v);
              }
            }
          });
      stream.Sync();
      t_unpack += grape::GetCurrentTime();
    }
    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(0);
    messages.RecordPackTime(0);
  }
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_MB_SERIAL_MSG_H_
