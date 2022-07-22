
#ifndef GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_PAGERANK_ASYNC_PAGERANK_WS_H_
#define GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_PAGERANK_ASYNC_PAGERANK_WS_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {

template <typename FRAG_T>
class AsyncPagerankWSContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using rank_t = float;

  explicit AsyncPagerankWSContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~AsyncPagerankWSContext() {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    value.D2H();
    delta.D2H();

    rank_t rank = 0, total_rank;
    rank_t local_delta = 0, total_delta;

    for (auto v : iv) {
      rank += value[v] / frag.GetTotalVerticesNum();
      local_delta += delta[v];
    }

    MPI_Reduce(&rank, &total_rank, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_delta, &total_delta, 1, MPI_FLOAT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (frag.fid() == 0) {
      LOG(INFO) << "Total rank: " << total_rank;
      LOG(INFO) << "Total delta: " << total_delta;
    }
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config,
            float damping_factor, double epslion) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->damping_factor = damping_factor;
    this->lb = app_config.lb;
    this->epslion = epslion;

    value.Init(vertices, 0);
    value.H2D();

    delta.Init(vertices);
    delta.SetValue(iv, 1 - damping_factor);
    delta.SetValue(ov, 0);
    delta.H2D();
    visited.Init(vertices);

    in_q.Init(iv.size());
    out_q_local.Init(iv.size());
    out_q_remote.Init(ov.size());

    auto recv_size = iv.size() * (sizeof(vid_t) + sizeof(rank_t)) * 7;
    auto send_size = ov.size() * (sizeof(vid_t) + sizeof(rank_t));

    messages.InitBuffer(send_size, recv_size);
    CHECK(lb != LoadBalancing::kStrict);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    value.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << value[v] / frag.GetTotalVerticesNum()
         << std::endl;
    }
  }

  rank_t damping_factor{};
  float epslion;
  LoadBalancing lb{};
  VertexArray<rank_t, vid_t> value;
  VertexArray<rank_t, vid_t> delta;
  Queue<vertex_t, uint32_t> in_q, out_q_local, out_q_remote;
  DenseVertexSet<vid_t> visited;
};

template <typename FRAG_T>
class AsyncPagerankWS
    : public GPUAppBase<FRAG_T, AsyncPagerankWSContext<FRAG_T>>,
      public ParallelEngine,
      public Communicator {
 public:
  INSTALL_GPU_WORKER(AsyncPagerankWS<FRAG_T>, AsyncPagerankWSContext<FRAG_T>,
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
    double t_compute = 0;

    t_compute -= grape::GetCurrentTime();
    ForEach(stream, ws_in,
            [=] __device__(vertex_t v) mutable { d_in_q.AppendWarp(v); });
    stream.Sync();
    t_compute += grape::GetCurrentTime();
    messages.ForceContinue();

    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(0);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_value = ctx.value.DeviceObject();
    auto d_delta = ctx.delta.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local = ctx.out_q_local;
    auto d_out_q_local = out_q_local.DeviceObject();
    auto& out_q_remote = ctx.out_q_remote;
    auto d_out_q_remote = out_q_remote.DeviceObject();
    auto& visited = ctx.visited;
    auto d_visited = visited.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto damping_factor = ctx.damping_factor;
    auto epslion = ctx.epslion;
    rank_t dangling_sum;
    auto total_vertices_num = frag.GetTotalVerticesNum();
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    t_unpack -= grape::GetCurrentTime();
    messages.template ParallelProcess<device_t, rank_t>(
        d_frag, [=] __device__(vertex_t v, rank_t msg) mutable {
          assert(d_frag.IsInnerVertex(v));
          if (atomicAdd(&d_delta[v], msg) + msg > epslion) {
            if (d_visited.Insert(v)) {
              d_in_q.AppendWarp(v);
            }
          }
        });
    stream.Sync();
    t_unpack += grape::GetCurrentTime();

    visited.Clear(stream);
    out_q_local.Clear(stream);
    out_q_remote.Clear(stream);

    auto in_size = in_q.size(stream);

    t_compute -= grape::GetCurrentTime();
    ForEachOutgoingEdge(
        stream, d_frag, WorkSourceArray<vertex_t>(in_q.data(), in_size),
        [=] __device__(vertex_t u) mutable -> rank_t {
          auto od = d_frag.GetLocalOutDegree(u);

          if (od > 0) {
            float delta = atomicExch(&d_delta[u], 0.0f);
            d_value[u] += delta;
            return damping_factor * delta / od;
          }
          return 0;
        },
        [=] __device__(const VertexMetadata<vid_t, rank_t>& u_and_rank,
                       const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          rank_t rank_to_send = u_and_rank.metadata;

          if (atomicAdd(&d_delta[v], rank_to_send) + rank_to_send > epslion) {
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

    rank_t local_dangling_sum = thrust::transform_reduce(
        thrust::cuda::par.on(stream.cuda_stream()),
        thrust::make_counting_iterator(iv.begin().GetValue()),
        thrust::make_counting_iterator(iv.end().GetValue()),
        [=] __device__(vid_t lid) mutable -> rank_t {
          vertex_t v(lid);
          auto od = d_frag.GetLocalOutDegree(v);
          rank_t delta = 0;

          if (od == 0) {
            delta = atomicExch(&d_delta[v], 0.0f);
            d_value[v] += delta;
          }
          return delta;
        },
        (rank_t) 0.0, thrust::plus<rank_t>());

    LOG(INFO) << "local_dangling_sum: " << local_dangling_sum;

    Sum(local_dangling_sum, dangling_sum);

    if (dangling_sum > 0) {
      ForEach(stream, WorkSourceRange<vertex_t>(iv.begin(), iv.size()),
              [=] __device__(vertex_t v) mutable {
                d_delta[v] +=
                    damping_factor * dangling_sum / total_vertices_num;
                if (d_delta[v] > epslion) {
                  if (d_visited.Insert(v)) {
                    d_out_q_local.AppendWarp(v);
                  }
                }
              });
    }
    stream.Sync();
    t_compute += grape::GetCurrentTime();

    t_pack -= grape::GetCurrentTime();
    WorkSourceArray<vertex_t> ws_out(out_q_remote.data(),
                                     out_q_remote.size(stream));

    messages.MakeOutput(
        stream, frag, WorkSourceRange<vertex_t>(ov.begin(), ov.size()),
        [=] __device__(vertex_t v) mutable {
          auto msg = thrust::make_pair(d_frag.GetOuterVertexGid(v), d_delta[v]);
          d_delta[v] = 0;
          return msg;
        });
    stream.Sync();
    t_pack += grape::GetCurrentTime();

    auto out_local_size = out_q_local.size(stream);
    VLOG(1) << "fid: " << frag.fid() << " In: " << in_size
            << " Local out: " << out_local_size
            << " Remote out: " << out_q_remote.size(stream);
    if (out_local_size > 0) {
      messages.ForceContinue();
    }

    in_q.Swap(out_q_local);
    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_EXAMPLES_ANALYTICAL_APPS_PAGERANK_ASYNC_PAGERANK_WS_H_
