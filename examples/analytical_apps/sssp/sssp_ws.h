#ifndef EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_WS_H_
#define EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_WS_H_
#include "VariadicTable.h"
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"
#include "grape_gpu/parallel/work_stealing_scheduler.h"

namespace grape_gpu {
template <typename FRAG_T>
class SSSPWSContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using dist_t = uint32_t;
  explicit SSSPWSContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->src_id = src_id;
    this->lb = app_config.lb;
    this->work_stealing = app_config.work_stealing;
    this->ws_k = app_config.ws_k;
    auto& comm_spec = messages.comm_spec();

    dist.Init(comm_spec, vertices, std::numeric_limits<dist_t>::max());
    dist.H2D();

    in_q.Init(iv.size());
    out_q_local.Init(iv.size());
    out_q_remote.Init(ov.size());

    // max number of vertex in a fragment
    size_t max_v_num = 0;

    for (int local_id = 0; local_id < comm_spec.local_num(); local_id++) {
      auto vertices_per_frag = frag.LocalVertices(local_id);

      max_v_num = std::max(max_v_num, vertices_per_frag.size());
    }
    remote_q.Init(max_v_num);
    remote_dist.resize(max_v_num);
    visited.Init(max_v_num);

    messages.InitBuffer(
        (sizeof(vid_t) + sizeof(dist_t)) * (ov.size() + max_v_num),
        (sizeof(vid_t) + sizeof(dist_t)) * iv.size());
    ws_scheduler = std::make_shared<WorkStealingScheduler<FRAG_T, dist_t>>(
        frag, messages.comm_spec());
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
  RemoteVertexArray<dist_t, vid_t> dist;
  Queue<vertex_t, uint32_t> in_q, out_q_local, out_q_remote;
  DenseVertexSet<vid_t> visited;

  Queue<vertex_t, uint32_t> remote_q;
  thrust::device_vector<dist_t> remote_dist;
  Counter counter;
  Counter counter_ws;

  std::shared_ptr<WorkStealingScheduler<FRAG_T, dist_t>> ws_scheduler;
  bool work_stealing;
  double ws_k;
};

template <typename FRAG_T>
class SSSPWS : public GPUAppBase<FRAG_T, SSSPWSContext<FRAG_T>>,
               public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(SSSPWS<FRAG_T>, SSSPWSContext<FRAG_T>, FRAG_T)
  using dist_t = typename context_t::dist_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using oid_t = typename fragment_t::oid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  // static constexpr bool need_build_device_vm = true;

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

    messages.RecordTime("Unpack", 0);
    messages.RecordTime("WS Scheduling", 0);
    messages.RecordTime("Remote Compute", 0);
    messages.RecordTime("Local Compute", 0);
    messages.RecordTime("Local Pack", 0);
    messages.RecordTime("Remote Pack", 0);
    messages.RecordTime("Pull Remote Value", 0);
    messages.RecordTime("Overhead", 0);

    messages.RecordCount("In", 0);
    messages.RecordCount("In af Stolen", 0);
    messages.RecordCount("Out Local", 0);
    messages.RecordCount("Out Remote", 0);
    messages.RecordCount("Stolen Frontier", 0);
    messages.RecordCount("Local #edge", 0);
    messages.RecordCount("Remote #edge", 0);
    //    messages.RecordCount("Active Local", 0);
    //    messages.RecordCount("Active Remote", 0);
  }

  int GetTotalDegree(Stream& stream, dev_fragment_t d_frag,
                     WorkSourceArray<vertex_t> frontier) {
    auto get_degree = [=] __host__ __device__(const vertex_t& v) -> size_t {
      return d_frag.GetLocalOutDegree(v);
    };
    thrust::device_vector<size_t> degree(frontier.size());
    thrust::transform_inclusive_scan(
        thrust::cuda::par.on(stream.cuda_stream()), frontier.data(),
        frontier.data() + frontier.size(), degree.begin(), get_degree,
        thrust::plus<size_t>());
    stream.Sync();

    return frontier.size() == 0 ? 0 : degree[frontier.size() - 1];
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto& stream = messages.stream();
    auto d_mm = messages.DeviceObject();
    auto& visited = ctx.visited;
    auto d_visited = visited.DeviceObject();
    auto& counter = ctx.counter;
    auto& counter_ws = ctx.counter_ws;

    // vars for local
    auto d_frag = frag.DeviceObject();
    auto iv = frag.InnerVertices();
    auto& dist = ctx.dist;
    auto d_dist = dist.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_local = ctx.out_q_local;
    auto d_out_q_local = out_q_local.DeviceObject();
    auto& out_q_remote = ctx.out_q_remote;
    auto d_out_q_remote = out_q_remote.DeviceObject();
    // vars for remote
    auto& ws_scheduler = *ctx.ws_scheduler;
    auto& remote_q = ctx.remote_q;
    auto d_remote_q = remote_q.DeviceObject();
    ArrayView<dist_t> d_remote_dist(ctx.remote_dist);
    double unpack_time{}, ws_scheduling_time{}, remote_compute_time{},
        remote_send_msg_time{}, local_compute_time{}, local_send_msg_time{},
        overhead_time{}, pull_remote_value_time{};

    unpack_time -= grape::GetCurrentTime();
    messages.template ParallelProcess<dev_fragment_t, dist_t>(
        d_frag, [=] __device__(vertex_t v, dist_t received_dist) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (received_dist < atomicMin(&d_dist[v], received_dist)) {
            if (d_visited.Insert(v)) {
              d_in_q.Append(v);
            }
          }
        });
    stream.Sync();
    unpack_time += grape::GetCurrentTime();

    size_t in_size = in_q.size(stream);
    size_t stolen_work_size = 0;
    size_t scanned_local_n_edges = 0;
    size_t scanned_remote_n_edges = 0;

    counter_ws.ResetAsync(stream);
    // N.B. DeviceObject can only be used after ResetAsync
    auto d_counter_ws = counter_ws.DeviceObject();

    if (ctx.work_stealing) {
      // report frontier and attached values to WSScheduler for workload
      // redistribution
      ws_scheduling_time -= grape::GetCurrentTime();
      ws_scheduler.ReportWork(stream, ArrayView<vertex_t>(in_q.data(), in_size),
                              ArrayView<dist_t>(d_dist.data(), d_dist.size()),
                              ctx.ws_k);
      stream.Sync();
      ws_scheduling_time += grape::GetCurrentTime();

      for (int local_id = 0; local_id < ws_scheduler.local_num(); local_id++) {
        if (local_id != ws_scheduler.local_id()) {
          // Get work source stolen from other GPUs and corresponding values
          auto ws = ws_scheduler.GetStolenWorkSource(local_id);
          auto attached_values = ws_scheduler.GetAttachedValues(local_id);
          // Get remote fragment object and its inner vertices
          auto r_frag = frag.DeviceObject(local_id);
          auto r_vertices = frag.LocalVertices(local_id);

          visited.Clear(stream);
          remote_q.Clear(stream);

          pull_remote_value_time -= grape::GetCurrentTime();
          // We firstly set d_remote_dist to max and copy the values of frontier
          // to corresponding cells just like a VertexArray
          thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()),
                         d_remote_dist.begin(), r_vertices.size(),
                         std::numeric_limits<dist_t>::max());
          ForEachWithIndex(stream, ws,
                           [=] __device__(size_t idx, vertex_t v) mutable {
                             d_remote_dist[v.GetValue()] = attached_values[idx];
                           });
          stream.Sync();
          pull_remote_value_time += grape::GetCurrentTime();

          // Now, visit edges, just like the process of normal SSSP
          // Actually, d_remote_dist and d_remote_q are allocated and accessed
          // on local, and later send to other GPUs as messages. We do tried to
          // allocate and access remotely, but we found that the overhead of
          // atomics is significantly high.
          remote_compute_time -= grape::GetCurrentTime();
          ForEachOutgoingEdge(
              stream, r_frag, ws,
              [=] __device__(vertex_t u) {
                return d_remote_dist[u.GetValue()];
              },
              [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                             const nbr_t& nbr) mutable {
                dist_t new_dist = metadata.metadata + nbr.get_data();
                vertex_t v = nbr.get_neighbor();

                if (new_dist <
                    atomicMin(&d_remote_dist[v.GetValue()], new_dist)) {
                  //                  d_counter_ws.Add();
                  if (d_visited.Insert(v)) {
                    d_remote_q.Append(v);
                  }
                }
              },
              ctx.lb);
          stream.Sync();
          remote_compute_time += grape::GetCurrentTime();

          overhead_time -= grape::GetCurrentTime();
          // For profiling
          scanned_remote_n_edges += GetTotalDegree(stream, r_frag, ws);
          stream.Sync();
          overhead_time += grape::GetCurrentTime();

          // Send messages to the "owner" of vertices
          remote_send_msg_time -= grape::GetCurrentTime();
          ForEach(
              stream,
              WorkSourceArray<vertex_t>(remote_q.data(), remote_q.size(stream)),
              [=] __device__(vertex_t v) mutable {
                auto fid = r_frag.GetFragId(v);
                d_mm.template SendToFragment(
                    fid, thrust::make_pair(r_frag.Vertex2Gid(v),
                                           d_remote_dist[v.GetValue()]));
              });
          stream.Sync();
          remote_send_msg_time += grape::GetCurrentTime();

          stolen_work_size += ws_scheduler.GetStolenWorkSource(local_id).size();
        }
      }
    }

    visited.Clear(stream);
    out_q_local.Clear(stream);
    out_q_remote.Clear(stream);

    counter.ResetAsync(stream);
    auto d_counter = counter.DeviceObject();

    auto ws = ws_scheduler.GetStolenWorkSource(ws_scheduler.local_id());

    stream.Sync();

    // Access local fragment and values
    local_compute_time -= grape::GetCurrentTime();
    ForEachOutgoingEdge(
        stream, d_frag, ws, [=] __device__(vertex_t u) { return d_dist[u]; },
        [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                       const nbr_t& nbr) mutable {
          dist_t new_dist = metadata.metadata + nbr.get_data();
          vertex_t v = nbr.get_neighbor();

          if (new_dist < atomicMin(&d_dist[v], new_dist)) {
            //              d_counter.Add();
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
    stream.Sync();
    local_compute_time += grape::GetCurrentTime();

    overhead_time -= grape::GetCurrentTime();
    // For profiling
    scanned_local_n_edges += GetTotalDegree(stream, d_frag, ws);
    stream.Sync();
    overhead_time += grape::GetCurrentTime();

    stream.Sync();

    auto out_local_len = out_q_local.size(stream);
    auto out_remote_len = out_q_remote.size(stream);

    // Send messages
    local_send_msg_time -= grape::GetCurrentTime();
    messages.MakeOutput(stream, frag,
                        WorkSourceArray<vertex_t>(out_q_remote.data(),
                                                  out_q_remote.size(stream)),
                        [=] __device__(vertex_t v) mutable {
                          return thrust::make_pair(d_frag.GetOuterVertexGid(v),
                                                   d_dist[v]);
                        });
    stream.Sync();
    local_send_msg_time += grape::GetCurrentTime();

    in_q.Swap(out_q_local);

    if (out_local_len + out_remote_len > 0) {
      messages.ForceContinue();
    }

    messages.RecordTime("Unpack", unpack_time);
    messages.RecordTime("WS Scheduling", ws_scheduling_time);
    messages.RecordTime("Remote Compute", remote_compute_time);
    messages.RecordTime("Local Compute", local_compute_time);
    messages.RecordTime("Local Pack", local_send_msg_time);
    messages.RecordTime("Remote Pack", remote_send_msg_time);
    messages.RecordTime("Pull Remote Value", pull_remote_value_time);
    messages.RecordTime("Profiling Overhead", overhead_time);

    messages.RecordCount("In", in_size);
    messages.RecordCount(
        "In af Stolen",
        ws_scheduler.GetStolenWorkSource(ws_scheduler.local_id()).size());
    messages.RecordCount("Out Local", out_local_len);
    messages.RecordCount("Out Remote", out_remote_len);
    messages.RecordCount("Stolen Frontier", stolen_work_size);
    messages.RecordCount("Local #edge", scanned_local_n_edges);
    messages.RecordCount("Remote #edge", scanned_remote_n_edges);
    //    messages.RecordCount("Active Local", counter.GetCount(stream));
    //    messages.RecordCount("Active Remote", counter_ws.GetCount(stream));
  }
};
}  // namespace grape_gpu

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_WS_H_
