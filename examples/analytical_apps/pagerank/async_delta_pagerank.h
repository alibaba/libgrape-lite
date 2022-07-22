#ifndef EXAMPLES_ANALYTICAL_APPS_ASYNCDELTAPAGERANK_ASYNCDELTAPAGERANK_H_
#define EXAMPLES_ANALYTICAL_APPS_ASYNCDELTAPAGERANK_ASYNCDELTAPAGERANK_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"
namespace grape_gpu {

// to simulate the async behavior
template <typename FRAG_T>
class AsyncDeltaPagerankContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using rank_t = float;

  explicit AsyncDeltaPagerankContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~AsyncDeltaPagerankContext() {
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    double tot_time = (get_msg_time + traversal_kernel_time + send_msg_time + 
                       mm->GetAccumulatedCommTime() + mm->GetSyncTime() + 
                       mm->GetStartARoundTime() + mm->GetBarrierTime());
    VLOG(1) << "# GPU " << dev << " runtime: " << tot_time * 1000 << " ms.";
    VLOG(1) << "# - unpack time: " << get_msg_time * 1000 << " ms.";
    VLOG(1) << "# - compute time: " << traversal_kernel_time * 1000 << " ms.";
    VLOG(1) << "# - pack time: " <<  send_msg_time * 1000 << " ms.";
    VLOG(1) << "# - comm time: " << mm->GetAccumulatedCommTime() * 1000 << " ms.";
    VLOG(1) << "# - sync time: " << mm->GetSyncTime() * 1000 << " ms.";
    VLOG(1) << "# - StartARound time: " << mm->GetStartARoundTime() * 1000 << " ms.";
    VLOG(1) << "# - barrier time: " << mm->GetBarrierTime() * 1000 << " ms.";
    VLOG(1) << "# - comp/total Ratio: " << traversal_kernel_time / tot_time;

    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    rank.D2H();

    rank_t local_rank = 0, total_rank;

    for (auto v : iv) {
      local_rank += rank[v];
    }
    MPI_Reduce(&local_rank, &total_rank, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (frag.fid() == 0) {
      LOG(INFO) << total_rank;
    }
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config,
            float damping_factor, std::string async_mode, double breakdown, int th,
            int max_color) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto total_vertices_num = frag.GetTotalVerticesNum();

    this->damping_factor = damping_factor;
    this->lb = app_config.lb;
    this->eps = 1e-9;
    this->breakdown = breakdown; // tunable
    this->th = th; // tunable
    this->max_color = max_color;
    this->color = 0;
    this->chunk_size = (frag.GetInnerVerticesNum() - 1) / max_color + 1;
    this->async_mode = async_mode=="Color"? 1:0;

    rank.Init(vertices, 0);
    rank.H2D();

    acc.Init(vertices, 0);
    acc.H2D();

    delta_rank.Init(vertices, (1 - damping_factor) / total_vertices_num);
    delta_rank.H2D();

    in_q.Init(iv.size());

    seen.Init(vertices, 0);
    seen.H2D();

    endure.Init(vertices, th);
    endure.H2D();

    remain.set(0);

    auto comm_vol_in_bytes =
        frag.OuterVertices().size() * (sizeof(vid_t) + sizeof(rank_t)) * 1.1;

    messages.InitBuffer(comm_vol_in_bytes, comm_vol_in_bytes);
    mm = &messages;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    rank.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << rank[v] << std::endl;
    }
  }

  rank_t damping_factor{};
  LoadBalancing lb{};
  VertexArray<rank_t, vid_t> rank;
  VertexArray<rank_t, vid_t> acc; // for simplicity
  VertexArray<rank_t, vid_t> delta_rank;
  VertexArray<int, vid_t> seen;
  VertexArray<int, vid_t> endure; // do not update if we updated in recent th iteration.
  Queue<vertex_t, vid_t> in_q;
  SharedValue<int> remain;
  int async_mode;
  int max_color;
  int color;
  int chunk_size;
  double breakdown;
  double eps;
  double th;
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
  GPUMessageManager* mm;
};

template <typename FRAG_T>
class AsyncDeltaPagerank : public GPUAppBase<FRAG_T, AsyncDeltaPagerankContext<FRAG_T>>,
                 public ParallelEngine,
                 public Communicator {
 public:
  INSTALL_GPU_WORKER(AsyncDeltaPagerank<FRAG_T>, AsyncDeltaPagerankContext<FRAG_T>, FRAG_T)
  using rank_t = typename context_t::rank_t;
  using device_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename device_t::vertex_t;
  using nbr_t = typename device_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto& stream = messages.stream();
    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_delta_rank = ctx.delta_rank.DeviceObject();
    auto d_acc = ctx.acc.DeviceObject();
    auto d_rank = ctx.rank.DeviceObject();
    auto d_seen = ctx.seen.DeviceObject();
    auto d_endure = ctx.endure.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto d_remain = ctx.remain.data();
    auto breakdown = ctx.breakdown;
    auto async_mode = ctx.async_mode;
    auto& color = ctx.color;
    auto chunk_size = ctx.chunk_size;
    auto max_color = ctx.max_color;
    auto eps = ctx.eps;
    auto th = ctx.th;
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto total_vertices_num = frag.GetTotalVerticesNum();
    auto damping_factor = ctx.damping_factor;

    ctx.get_msg_time -= grape::GetCurrentTime();
    messages.template ParallelProcess<device_t, rank_t>(
        d_frag, [=] __device__(vertex_t v, rank_t msg) mutable {
          assert(d_frag.IsInnerVertex(v));
          atomicAdd(&d_delta_rank[v], msg);
        });
    ctx.get_msg_time += grape::GetCurrentTime();

    auto traversal_kernel_time = grape::GetCurrentTime();
    stream.Sync();

    // Scan
    WorkSourceRange<vertex_t> ws_in_range(iv.begin(), iv.size());
    ForEach(stream, ws_in_range,
      [=] __device__(vertex_t v) mutable {
        d_delta_rank[v] += d_acc[v];
        d_acc[v] = 0;
        if (d_delta_rank[v] > eps){
          if (cond(async_mode, v, d_delta_rank[v], breakdown, d_endure[v],
                   color, max_color, chunk_size)) {
            d_in_q.AppendWarp(v);
          // clean delta after expansion
          }
          // enable FroceContinue;
          *d_remain = 1; // benign;
        }
        d_endure[v] --;
      }
    );

    if(ctx.remain.get(stream)) messages.ForceContinue();
    ctx.remain.set(0, stream);
    color = (color + 1) % max_color;

    // Expand
    WorkSourceArray<vertex_t> ws_in(in_q.data(), in_q.size(stream));
    ForEachOutgoingEdge(
        stream, d_frag, ws_in,
        [=] __device__(vertex_t u) -> rank_t {
          rank_t rank_send =
              damping_factor * d_delta_rank[u] / d_frag.GetLocalOutDegree(u);
          return rank_send;
        },
        [=] __device__(const VertexMetadata<vid_t, rank_t>& u_and_rank,
                       const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          rank_t rank_to_send = u_and_rank.metadata;
          d_seen[v] = 1; // benign
          atomicAdd(&d_acc[v], rank_to_send);
        },
        ctx.lb
    );

    // Update
    ForEach(stream, ws_in,
      [=] __device__(vertex_t v) mutable{
        d_rank[v] += d_delta_rank[v];
        d_delta_rank[v] = 0;
        d_endure[v] = th;
      }
    );
    in_q.Clear(stream);

    stream.Sync();
    traversal_kernel_time = grape::GetCurrentTime() - traversal_kernel_time;
    ctx.traversal_kernel_time += traversal_kernel_time;


    ctx.send_msg_time -= grape::GetCurrentTime();
    for (fid_t fid = 0; fid < frag.fnum(); fid++) {
      ov = frag.OuterVertices(fid);
      ws_in_range = WorkSourceRange<vertex_t>(ov.begin(), ov.size());

      ForEach(stream, ws_in_range, [=] __device__(vertex_t v) mutable {
        if (d_seen[v]) {
          d_mm.template SyncStateOnOuterVertexWarpOpt(d_frag, v,
                                                      d_acc[v]);
          d_acc[v] = 0;
        }
        d_seen[v] = 0;
      });
    }
    stream.Sync();
    ctx.send_msg_time += grape::GetCurrentTime();

    VLOG(1) << "Frag " << frag.fid()
            << " Kernel time: " << traversal_kernel_time * 1000;
  }

  inline __device__
  bool cond(int async_mode, vertex_t v, double v_delta, double breakdown,
            double v_endure, int color, int max_color, int chunk_size){
    if (async_mode==0) { // endure-based
      if (v_delta > breakdown || v_endure <= 0) return true;
      else return false;
    } else { // chunk-based
      if(v.GetValue() >= color * chunk_size &&
         v.GetValue() < (color+1) * chunk_size) return true;
      else return false;
    }
  }
};
}  // namespace grape_gpu
#endif
