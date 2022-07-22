#ifndef EXAMPLES_ANALYTICAL_APPS_BCPAGERANK_BCPAGERANK_H_
#define EXAMPLES_ANALYTICAL_APPS_BCPAGERANK_BCCPAGERANK_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
template <typename FRAG_T>
class BCPagerankContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using rank_t = float;

  explicit BCPagerankContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~BCPagerankContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "AsyncPagerank kernel time: " << traversal_kernel_time * 1000;
    VLOG(1) << "Send msg time: " << send_msg_time * 1000;
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    LOG(INFO) << "GPU " << dev << " Compute time: " << compute_time * 1000
              << " ms Comm time: " << mm->GetAccumulatedCommTime() * 1000
              << " ms Ratio: " << compute_time / mm->GetAccumulatedCommTime();
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    value.D2H();

    rank_t rank = 0, total_rank;

    for (auto v : iv) {
      rank += value[v] / frag.GetTotalVerticesNum();
    }
    MPI_Reduce(&rank, &total_rank, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (frag.fid() == 0) {
      LOG(INFO) << total_rank;
    }
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config,
            float damping_factor, double epslion) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto total_vertices_num = frag.GetTotalVerticesNum();

    this->damping_factor = damping_factor;
    this->lb = app_config.lb;
    this->epslion = epslion;  // tunable

    value.Init(vertices, 0);
    value.H2D();

    delta.Init(vertices);
    delta.SetValue(iv, 1 - damping_factor);
    delta.SetValue(ov, 0);
    delta.H2D();

    auto capacity = frag.GetEdgeNum() * 0.5;

    in_q.Init(iv);
    out_q_local.Init(iv);
    out_q_remote.Init(ov);
    tmp_q.Init(vertices.size());

    auto comm_vol_in_bytes = capacity * (sizeof(vid_t) + sizeof(rank_t)) * 1.1;

    messages.InitBuffer(comm_vol_in_bytes, comm_vol_in_bytes);
    mm = &messages;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    value.D2H();

    //    for (auto v : iv) {
    //      os << frag.GetId(v) << " " << value[v] / frag.GetTotalVerticesNum()
    //         << std::endl;
    //    }
  }

  rank_t damping_factor{};
  float epslion;
  LoadBalancing lb{};
  Queue<vertex_t> tmp_q;
  VertexArray<rank_t, vid_t> value;
  VertexArray<rank_t, vid_t> delta;
  DenseVertexSet<vid_t> in_q, out_q_local, out_q_remote;
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
  double compute_time{};
  GPUMessageManager* mm;
};

template <typename FRAG_T>
class BCPagerank : public GPUAppBase<FRAG_T, BCPagerankContext<FRAG_T>>,
                   public ParallelEngine,
                   public Communicator {
 public:
  INSTALL_GPU_WORKER(BCPagerank<FRAG_T>, BCPagerankContext<FRAG_T>, FRAG_T)
  using rank_t = typename context_t::rank_t;
  using device_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename device_t::vertex_t;
  using nbr_t = typename device_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());

    ForEach(stream, ws_in,
            [=] __device__(vertex_t v) mutable { d_in_q.Insert(v); });
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_value = ctx.value.DeviceObject();
    auto d_delta = ctx.delta.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& in_q = ctx.in_q;
    auto& out_q_local = ctx.out_q_local;
    auto& out_q_remote = ctx.out_q_remote;
    auto& tmp_q = ctx.tmp_q;
    auto d_tmp_q = tmp_q.DeviceObject();
    auto d_in_q = in_q.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto damping_factor = ctx.damping_factor;
    auto epslion = ctx.epslion;

    ctx.get_msg_time -= grape::GetCurrentTime();
    messages.template ParallelProcess<device_t, rank_t>(
        d_frag, [=] __device__(vertex_t v, rank_t msg) mutable {
          assert(d_frag.IsInnerVertex(v));
          if (atomicAdd(&d_delta[v], msg) + msg > epslion) {
            d_in_q.Insert(v);
          }
        });
    ctx.get_msg_time += grape::GetCurrentTime();

    auto traversal_kernel_time = grape::GetCurrentTime();

    stream.Sync();

    ctx.compute_time -= grape::GetCurrentTime();

    while (in_q.Count(stream) > 0) {
      auto d_in_q = in_q.DeviceObject();
      // Bitmap to queue
      {
        WorkSourceRange<vertex_t> ws_in(iv.begin(), iv.size());

        tmp_q.Clear(stream);
        ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
          if (d_in_q.Exist(v)) {
            d_tmp_q.AppendWarp(v);
          }
        });
      }

      WorkSourceArray<vertex_t> ws_in(tmp_q.data(), tmp_q.size(stream));

      CHECK(ctx.lb != LoadBalancing::kStrict);

      auto d_out_q_local = out_q_local.DeviceObject();
      auto d_out_q_remote = out_q_remote.DeviceObject();

      // Expand
      ForEachOutgoingEdge(
          stream, d_frag, ws_in,
          [=] __device__(vertex_t u) mutable -> rank_t {
            float delta = atomicExch(&d_delta[u], 0.0f);

            d_value[u] += delta;

            return damping_factor * delta / d_frag.GetLocalOutDegree(u);
          },
          [=] __device__(const VertexMetadata<vid_t, rank_t>& u_and_rank,
                         const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            rank_t rank_to_send = u_and_rank.metadata;

            if (atomicAdd(&d_delta[v], rank_to_send) + rank_to_send > epslion) {
              if (d_frag.IsInnerVertex(v)) {
                d_out_q_local.Insert(v);
              } else {
                d_out_q_remote.Insert(v);
              }
            }
          },
          ctx.lb);

      VLOG(10) << "In: " << in_q.Count() << " out: " << out_q_local.Count()
               << " remote: " << out_q_remote.Count();

      in_q.Clear(stream);
      in_q.Swap(out_q_local);
    }

    stream.Sync();
    ctx.compute_time += grape::GetCurrentTime();

    RangeMarker marker(true, "Range1");
    auto d_out_q_remote = out_q_remote.DeviceObject();
    // Send message
    for (fid_t fid = 0; fid < frag.fnum(); fid++) {
      ov = frag.OuterVertices(fid);
      WorkSourceRange<vertex_t> ws_in(ov.begin(), ov.size());

      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_out_q_remote.Exist(v) && d_delta[v] > 0) {
          d_mm.template SyncStateOnOuterVertexWarpOpt(d_frag, v, d_delta[v]);
          d_delta[v] = 0;
        }
      });
    }
    out_q_remote.Clear(stream);

    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    ctx.traversal_kernel_time += traversal_kernel_time;

    VLOG(1) << "Frag " << frag.fid()
            << " Kernel time: " << traversal_kernel_time * 1000;
    marker.Stop();
  }
};
}  // namespace grape_gpu
#endif
