#ifndef EXAMPLES_ANALYTICAL_APPS_ASYNCPAGERANK_ASYNCPAGERANK_H_
#define EXAMPLES_ANALYTICAL_APPS_ASYNCPAGERANK_ASYNCPAGERANK_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {

// to simulate the async behavior
template <typename FRAG_T>
class AsyncPagerankContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using rank_t = float;

  explicit AsyncPagerankContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~AsyncPagerankContext() {
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    double tot_time = (get_msg_time + traversal_kernel_time + send_msg_time +
                       mm->GetAccumulatedCommTime() + mm->GetSyncTime() +
                       mm->GetStartARoundTime() + mm->GetBarrierTime());
    VLOG(1) << "# GPU " << dev << " runtime: " << tot_time * 1000 << " ms.";
    VLOG(1) << "# - unpack time: " << get_msg_time * 1000 << " ms.";
    VLOG(1) << "# - compute time: " << traversal_kernel_time * 1000 << " ms.";
    VLOG(1) << "# - pack time: " << send_msg_time * 1000 << " ms.";
    VLOG(1) << "# - comm time: " << mm->GetAccumulatedCommTime() * 1000
            << " ms.";
    VLOG(1) << "# - sync time: " << mm->GetSyncTime() * 1000 << " ms.";
    VLOG(1) << "# - StartARound time: " << mm->GetStartARoundTime() * 1000
            << " ms.";
    VLOG(1) << "# - barrier time: " << mm->GetBarrierTime() * 1000 << " ms.";
    VLOG(1) << "# - comp/total Ratio: " << traversal_kernel_time / tot_time;

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
            float damping_factor, double epslion, int max_color) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto total_vertices_num = frag.GetTotalVerticesNum();

    this->damping_factor = damping_factor;
    this->lb = app_config.lb;
    this->chunk_size = round_up(frag.GetInnerVerticesNum(), max_color);
    this->max_round = max_color;
    this->epslion = epslion;  // tunable

    value.Init(vertices, 0);
    value.H2D();

    delta.Init(vertices);
    delta.SetValue(iv, 1 - damping_factor);
    delta.SetValue(ov, 0);
    delta.H2D();

    auto capacity = frag.GetEdgeNum() * 1.0;

    in_q.Init(vertices);
    out_q.Init(vertices);
    tmp_q.Init(vertices.size());

    auto recv_size = iv.size() * (sizeof(vid_t) + sizeof(rank_t)) * 7;
    auto send_size = ov.size() * (sizeof(vid_t) + sizeof(rank_t));

    messages.InitBuffer(send_size, recv_size);
    mm = &messages;
    LOG(INFO) << "chunk_size: " << chunk_size;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    value.D2H();

    //    for (auto v : iv) {
    //      os << frag.GetId(v) << " " << value[v] / frag.GetTotalVerticesNum()
    //         << std::endl;
    //    }

    for (auto& line : active_info) {
      os << line << std::endl;
    }
  }

  rank_t damping_factor{};
  float epslion;
  LoadBalancing lb{};
  Queue<vertex_t> tmp_q;
  VertexArray<rank_t, vid_t> value;
  VertexArray<rank_t, vid_t> delta;
  DenseVertexSet<vid_t> in_q, out_q;
  int chunk_size;
  int round{};
  int max_round{};
  double compute_time{};
  GPUMessageManager* mm;
  std::vector<std::string> active_info;

  double th;
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
};

template <typename FRAG_T>
class AsyncPagerank : public GPUAppBase<FRAG_T, AsyncPagerankContext<FRAG_T>>,
                      public ParallelEngine,
                      public Communicator {
 public:
  INSTALL_GPU_WORKER(AsyncPagerank<FRAG_T>, AsyncPagerankContext<FRAG_T>,
                     FRAG_T)
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

    in_q.Clear(stream);
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
    auto& out_q = ctx.out_q;
    auto& tmp_q = ctx.tmp_q;
    auto d_in_q = in_q.DeviceObject();
    auto d_out_q = out_q.DeviceObject();
    auto d_tmp_q = tmp_q.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto damping_factor = ctx.damping_factor;
    auto epslion = ctx.epslion;
    auto& round = ctx.round;

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

    // count after 20 rounds
    round++;

    WorkSourceArray<vertex_t> ws_in(tmp_q.data(), tmp_q.size(stream));

    CHECK(ctx.lb != LoadBalancing::kStrict);

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
            d_out_q.Insert(v);
          }
        },
        ctx.lb);

    stream.Sync();

    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    ctx.traversal_kernel_time += traversal_kernel_time;

    ctx.send_msg_time -= grape::GetCurrentTime();
    RangeMarker marker(true, "Range1");
    // Send message
    for (fid_t fid = 0; fid < frag.fnum(); fid++) {
      ov = frag.OuterVertices(fid);
      WorkSourceRange<vertex_t> ws_in(ov.begin(), ov.size());

      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_out_q.Exist(v) && d_delta[v] > 0) {
          d_mm.template SyncStateOnOuterVertexWarpOpt(d_frag, v, d_delta[v]);
          d_delta[v] = 0;
        }
      });
    }

    auto out_count = out_q.Count(stream);
    VLOG(2) << "Frag: " << frag.fid() << " Color: " << round
            << " In: " << ws_in.size() << " out local: " << out_count;

    if (out_count > 0) {
      messages.ForceContinue();
    }

    in_q.Clear(stream);
    in_q.Swap(out_q);
    stream.Sync();
    ctx.send_msg_time += grape::GetCurrentTime();

    VLOG(1) << "Frag " << frag.fid()
            << " Kernel time: " << traversal_kernel_time * 1000;
    marker.Stop();
  }
};
}  // namespace grape_gpu
#endif
