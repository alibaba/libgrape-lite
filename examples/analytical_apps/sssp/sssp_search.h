#ifndef EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SEARCH_H_
#define EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SEARCH_H_
#include "VariadicTable.h"
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"
#include "grape_gpu/parallel/work_stealing_scheduler.h"

// This application is used for searching the cost matrix $\mathcal{C}$
namespace grape_gpu {
template <typename FRAG_T>
class SSSPSEARCHContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
#ifdef FLOAT_WEIGHT
  using dist_t = float;
#else
  using dist_t = uint32_t;
#endif
  explicit SSSPSEARCHContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~SSSPSEARCHContext() {
    std::stringstream ss;
    ss << "Summary:" << std::endl;
    PrintWorkTimeStat(ss, get_msg_time_, ws_scheduling_time_,
                      local_compute_time_, remote_compute_time_,
                      local_send_msg_time_, remote_send_msg_time_);
    if (ws_scheduler->local_id() == 0) {
      LOG(INFO) << ss.str();
    }
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id,
            int init_prio, int mr) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->src_id = src_id;
    this->lb = app_config.lb;
    this->work_stealing = app_config.work_stealing;
    this->ws_k = app_config.ws_k;
    this->mr = mr;
    auto& comm_spec = messages.comm_spec();

    dist_curr.Init(comm_spec, vertices, std::numeric_limits<dist_t>::max());
    dist_curr.H2D();
    dist_next.Init(comm_spec, vertices, std::numeric_limits<dist_t>::max());
    dist_next.H2D();

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
    remote_dist_curr.resize(max_v_num);
    remote_dist_next.resize(max_v_num);
    visited.Init(max_v_num);

    double weight_sum = 0;

    for (auto v : iv) {
      auto oes = frag.GetOutgoingAdjList(v);
      for (auto& e : oes) {
        weight_sum += e.get_data();
      }
    }
    if (init_prio == 0) {
      /**
       * We select a similar heuristic, Δ = cw/d,
          where d is the average degree in the graph, w is the average
          edge weight, and c is the warp width (32 on our GPUs)
          Link: https://people.csail.mit.edu/jshun/papers/DBGO14.pdf
       */
      init_prio = 32 * (weight_sum / frag.GetEdgeNum()) /
                  (1.0 * frag.GetEdgeNum() / iv.size());
    }
    prio = init_prio;

    messages.InitBuffer(
        (sizeof(vid_t) + sizeof(dist_t)) * (ov.size() + max_v_num),
        (sizeof(vid_t) + sizeof(dist_t)) * iv.size());
    mm = &messages;
    ws_scheduler = std::make_shared<WorkStealingScheduler<FRAG_T, dist_t>>(
        frag, messages.comm_spec());
    int m = comm_spec.local_num();
    real_times.resize(m);
    for(int i=0; i<m; ++i) real_times[i].resize(m);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    dist_curr.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << dist_curr[v] << std::endl;
    }
  }

  void PrintWorkSizeStat(std::stringstream& ss, size_t in_size, size_t ws_size,
                         size_t out_local_len, size_t out_remote_len,
                         size_t stolen_work_size, size_t scanned_local_n_edges,
                         size_t scanned_remote_n_edges, size_t active_nv,
                         size_t active_remote_nv) {
    std::vector<size_t> size_array{
        in_size,
        ws_size,
        out_local_len,
        out_remote_len,
        stolen_work_size,
        scanned_local_n_edges,
        scanned_remote_n_edges,
        scanned_local_n_edges + scanned_remote_n_edges,
        active_nv,
        active_remote_nv};
    if (ws_scheduler->local_id() == 0) {
      VariadicTable<fid_t, size_t, size_t, size_t, size_t, size_t, size_t,
                    size_t, size_t, size_t, size_t>
          vt_work({"fid", "In", "In WS", "Out Local", "Out Remote",
                   "Stolen size", "Local #edge", "Remote #edge", "#edge",
                   "Active", "Active Remote"});
      std::vector<size_t> all_size_array(size_array.size() *
                                         ws_scheduler->local_num());

      MPI_Gather(size_array.data(), size_array.size(), my_MPI_SIZE_T,
                 all_size_array.data(), size_array.size(), my_MPI_SIZE_T, 0,
                 ws_scheduler->local_comm());

      for (int local_id = 0; local_id < ws_scheduler->local_num(); local_id++) {
        vt_work.addRow(local_id,
                       all_size_array[local_id * size_array.size() + 0],
                       all_size_array[local_id * size_array.size() + 1],
                       all_size_array[local_id * size_array.size() + 2],
                       all_size_array[local_id * size_array.size() + 3],
                       all_size_array[local_id * size_array.size() + 4],
                       all_size_array[local_id * size_array.size() + 5],
                       all_size_array[local_id * size_array.size() + 6],
                       all_size_array[local_id * size_array.size() + 7],
                       all_size_array[local_id * size_array.size() + 8],
                       all_size_array[local_id * size_array.size() + 9]);
      }
      vt_work.template print(ss);
    } else {
      MPI_Gather(size_array.data(), size_array.size(), my_MPI_SIZE_T, nullptr,
                 0, my_MPI_SIZE_T, 0, ws_scheduler->local_comm());
    }
  }

  void CollectRealTime(){
    int id = ws_scheduler->local_id();
    int sz = ws_scheduler->local_num();
    if(id == 0){
      std::vector<double> all_time_array(real_times[id].size() * sz);
      MPI_Gather(real_times[id].data(), real_times[id].size(), MPI_DOUBLE,
                 all_time_array.data(), real_times[id].size(), MPI_DOUBLE, 0,
                 ws_scheduler->local_comm());
      for(int i=0; i<sz; ++i) {
        for(int j=0; j<sz; ++j) {
          real_times[j][i] = all_time_array[i*sz + j];
        }
      }
    } else {
      MPI_Gather(real_times[id].data(), real_times[id].size(), MPI_DOUBLE, nullptr, 0,
                 MPI_DOUBLE, 0, ws_scheduler->local_comm());
    }
  }

#ifdef PROFILING
  void PrintWorkTimeStat(std::stringstream& ss, double get_msg_time,
                         double ws_scheduling_time, double local_compute_time,
                         double remote_compute_time, double local_send_msg_time,
                         double remote_send_msg_time) {
    std::vector<double> time_array{get_msg_time,        ws_scheduling_time,
                                   local_compute_time,  remote_compute_time,
                                   local_send_msg_time, remote_send_msg_time};

    if (ws_scheduler->local_id() == 0) {
      std::vector<double> all_time_array(time_array.size() *
                                         ws_scheduler->local_num());

      VariadicTable<fid_t, double, double, double, double, double, double>
          vt_time({"fid", "GetMsg", "WS Sch", "LocalComp", "RemoteComp",
                   "LocalSend", "RemoteSend"});
      MPI_Gather(time_array.data(), time_array.size(), MPI_DOUBLE,
                 all_time_array.data(), time_array.size(), MPI_DOUBLE, 0,
                 ws_scheduler->local_comm());

      vt_time.setColumnPrecision({1, 4, 4, 4, 4, 4, 4});
      for (int local_id = 0; local_id < ws_scheduler->local_num(); local_id++) {
        vt_time.addRow(local_id,
                       all_time_array[local_id * time_array.size() + 0],
                       all_time_array[local_id * time_array.size() + 1],
                       all_time_array[local_id * time_array.size() + 2],
                       all_time_array[local_id * time_array.size() + 3],
                       all_time_array[local_id * time_array.size() + 4],
                       all_time_array[local_id * time_array.size() + 5]);
      }
      vt_time.template print(ss);
    } else {
      MPI_Gather(time_array.data(), time_array.size(), MPI_DOUBLE, nullptr, 0,
                 MPI_DOUBLE, 0, ws_scheduler->local_comm());
    }
  }
#endif

  bool accept() {
    double min_= std::numeric_limits<dist_t>::max();
    double max_=0;
    bool to_process = false;
    if(ws_scheduler->local_id() == 0) {
      for(int i=0; i<real_times.size(); ++i) {
        double sum_ = 0;
        for(int j=0; j<real_times[i].size(); ++j) {
          sum_ += real_times[j][i];
        }
        min_ = std::min(min_, sum_);
        max_ = std::max(max_, sum_);
      }
      LOG(INFO) << "judging " << min_ << " " <<  max_;
      if((max_-min_)/max_ < 0.1) to_process = true;
      if(max_*1000 <= 10.0) to_process = true;
    }
    MPI_Bcast(&to_process, 1, MPI_CHAR, 0, ws_scheduler->local_comm());
    return to_process;
  }

  oid_t src_id;
  LoadBalancing lb{};
  RemoteVertexArray<dist_t, vid_t> dist_curr;
  RemoteVertexArray<dist_t, vid_t> dist_next;
  Queue<vertex_t, uint32_t> in_q, out_q_local, out_q_remote;
  DenseVertexSet<vid_t> visited;

  Queue<vertex_t, uint32_t> remote_q;
  thrust::device_vector<dist_t> remote_dist_curr;
  thrust::device_vector<dist_t> remote_dist_next;
  Counter counter;
  Counter counter_ws;

  dist_t init_prio{};
  dist_t prio{};
  double get_msg_time_{}, ws_scheduling_time_{}, remote_compute_time_{},
      remote_send_msg_time_{}, local_compute_time_{}, local_send_msg_time_{};

  GPUMessageManager* mm;
  std::shared_ptr<WorkStealingScheduler<FRAG_T, dist_t>> ws_scheduler;
  bool work_stealing;
  double ws_k;
  int mr;
  int curr_round{};
  std::vector<std::vector<double>> real_times;
};

template <typename FRAG_T>
class SSSPSEARCH : public GPUAppBase<FRAG_T, SSSPSEARCHContext<FRAG_T>>,
               public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(SSSPSEARCH<FRAG_T>, SSSPSEARCHContext<FRAG_T>, FRAG_T)
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
          frag.DeviceObject(), ctx.dist_curr.DeviceObject(),
          ctx.in_q.DeviceObject());
    }
    messages.ForceContinue();
    messages.RecordUnpackTime(0.0);
    messages.RecordComputeTime(0.0);
    messages.RecordPackTime(0.0);
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
    auto& dist_curr = ctx.dist_curr;
    auto d_dist_curr = dist_curr.DeviceObject();
    auto& dist_next = ctx.dist_next;
    auto d_dist_next = dist_next.DeviceObject();
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
    ArrayView<dist_t> d_remote_dist_curr(ctx.remote_dist_curr);
    ArrayView<dist_t> d_remote_dist_next(ctx.remote_dist_next);
    double get_msg_time{}, ws_scheduling_time{}, remote_compute_time{},
        remote_send_msg_time{}, local_compute_time{}, local_send_msg_time{};

    get_msg_time = grape::GetCurrentTime();
    messages.template ParallelProcess<dev_fragment_t, dist_t>(
        d_frag, [=] __device__(vertex_t v, dist_t received_dist) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (received_dist < atomicMin(&d_dist_curr[v], received_dist)) {
            atomicMin(&d_dist_next[v], received_dist);
            if (d_visited.Insert(v)) {
              d_in_q.Append(v);
            }
          }
        });
    stream.Sync();
    get_msg_time = (grape::GetCurrentTime() - get_msg_time) * 1000;

    //if (ctx.curr_round++ == ctx.mr) {
    //  return;
    //}

    size_t in_size = in_q.size(stream);
    size_t stolen_work_size = 0;
    size_t scanned_local_n_edges = 0;
    size_t scanned_remote_n_edges = 0;

    counter_ws.ResetAsync(stream);
    // N.B. DeviceObject can only be used after ResetAsync
    auto d_counter_ws = counter_ws.DeviceObject();

    if (ctx.work_stealing) {
      ws_scheduling_time = grape::GetCurrentTime();
      ws_scheduler.Updatebw(ctx.real_times);
      ws_scheduler.ReportWork(
          stream, ArrayView<vertex_t>(in_q.data(), in_size),
          ArrayView<dist_t>(d_dist_curr.data(), d_dist_curr.size()), ctx.ws_k);
      stream.Sync();
      ws_scheduling_time =
          (grape::GetCurrentTime() - ws_scheduling_time) * 1000;

      for (int local_id = 0; local_id < ws_scheduler.local_num(); local_id++) {
        if (local_id != ws_scheduler.local_id()) {
          auto ws = ws_scheduler.GetStolenWorkSource(local_id);
          auto attached_values = ws_scheduler.GetAttachedValues(local_id);
          auto r_frag = frag.DeviceObject(local_id);
          auto r_vertices = frag.LocalVertices(local_id);

          visited.Clear(stream);
          remote_q.Clear(stream);

          // reset remote dist
//          thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()),
//                         d_remote_dist_curr.begin(), r_vertices.size(),
//                         std::numeric_limits<dist_t>::max());

          // fill remote dist with value attached with frontier
          ForEachWithIndex(
              stream, ws, [=] __device__(size_t idx, vertex_t v) mutable {
                d_remote_dist_curr[v.GetValue()] = attached_values[idx];
              });

          if (ws.size() / frag.GetVerticesNum() > 0.1) {
            auto d_remote_dist_curr_src = ctx.dist_curr.DeviceObject(local_id);
            thrust::copy(
                thrust::cuda::par.on(stream.cuda_stream()),
                d_remote_dist_curr_src.data(),
                d_remote_dist_curr_src.data() + d_remote_dist_curr_src.size(),
                d_remote_dist_next.begin());
          } else {
            thrust::fill_n(thrust::cuda::par.on(stream.cuda_stream()),
                           d_remote_dist_next.begin(), r_vertices.size(),
                           std::numeric_limits<dist_t>::max());
          }

          stream.Sync();
          remote_compute_time -= grape::GetCurrentTime();
          double single_remote_compute_time = grape::GetCurrentTime();
          ForEachOutgoingEdge(
              stream, r_frag, ws,
              [=] __device__(vertex_t u) {
                return d_remote_dist_curr[u.GetValue()];
              },
              [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                             const nbr_t& nbr) mutable {
                dist_t new_dist = metadata.metadata + nbr.get_data();
                vertex_t v = nbr.get_neighbor();

                if (new_dist <
                    atomicMin(&d_remote_dist_next[v.GetValue()], new_dist)) {
                  d_counter_ws.Add();
                  if (d_visited.Insert(v)) {
                    d_remote_q.Append(v);
                  }
                }
              },
              ctx.lb);
          stream.Sync();
          remote_compute_time += grape::GetCurrentTime();
          ctx.real_times[ws_scheduler.local_id()][local_id] = grape::GetCurrentTime() - single_remote_compute_time;
          scanned_remote_n_edges += GetTotalDegree(stream, r_frag, ws);

          stream.Sync();

          remote_send_msg_time -= grape::GetCurrentTime();
          ForEach(
              stream,
              WorkSourceArray<vertex_t>(remote_q.data(), remote_q.size(stream)),
              [=] __device__(vertex_t v) mutable {
                auto fid = r_frag.GetFragId(v);
                d_mm.template SendToFragment(
                    fid, thrust::make_pair(r_frag.Vertex2Gid(v),
                                           d_remote_dist_next[v.GetValue()]));
              });
          stream.Sync();
          remote_send_msg_time += grape::GetCurrentTime();

          stolen_work_size += ws_scheduler.GetStolenWorkSource(local_id).size();
        }
      }
    }

    remote_compute_time *= 1000;
    remote_send_msg_time *= 1000;

    visited.Clear(stream);
    out_q_local.Clear(stream);
    out_q_remote.Clear(stream);

    counter.ResetAsync(stream);
    auto d_counter = counter.DeviceObject();

    if (ctx.work_stealing) {
      auto ws = ws_scheduler.GetStolenWorkSource(ws_scheduler.local_id());

      stream.Sync();

      local_compute_time -= grape::GetCurrentTime();
      double single_local_compute_time = grape::GetCurrentTime();
      ForEachOutgoingEdge(
          stream, d_frag, ws,
          [=] __device__(vertex_t u) { return d_dist_curr[u]; },
          [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                         const nbr_t& nbr) mutable {
            dist_t new_dist = metadata.metadata + nbr.get_data();
            vertex_t v = nbr.get_neighbor();

            if (new_dist < atomicMin(&d_dist_next[v], new_dist)) {
              d_counter.Add();
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
      ctx.real_times[ws_scheduler.local_id()][ws_scheduler.local_id()] = grape::GetCurrentTime() - single_local_compute_time;

      scanned_local_n_edges += GetTotalDegree(stream, d_frag, ws);
      stream.Sync();
    } else {
      // process rest of the works
      auto bounds = ws_scheduler.CalculateBounds(
          stream, ArrayView<vertex_t>(in_q.data(), in_size), ctx.ws_k);

      std::stringstream ss;

      ss << "fid: " << ws_scheduler.local_id() << " ";

      std::sort(bounds.begin(), bounds.end(),
                [](const thrust::pair<size_t, size_t>& e1,
                   const thrust::pair<size_t, size_t>& e2) {
                  return e1.first < e2.first;
                });

      for (auto& bound : bounds) {
        auto len = bound.second - bound.first;
        auto ws = WorkSourceArray<vertex_t>(in_q.data() + bound.first, len);

        ss << "[" << bound.first << ", " << bound.second << ") ";

        stream.Sync();

        local_compute_time -= grape::GetCurrentTime();
        ForEachOutgoingEdge(
            stream, d_frag, ws,
            [=] __device__(vertex_t u) { return d_dist_curr[u]; },
            [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                           const nbr_t& nbr) mutable {
              dist_t new_dist = metadata.metadata + nbr.get_data();
              vertex_t v = nbr.get_neighbor();

              if (new_dist < atomicMin(&d_dist_next[v], new_dist)) {
                d_counter.Add();
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

        scanned_local_n_edges += GetTotalDegree(stream, d_frag, ws);
      }
      LOG(INFO) << ss.str();
    }
    local_compute_time *= 1000;

    stream.Sync();

    auto out_local_len = out_q_local.size(stream);
    auto out_remote_len = out_q_remote.size(stream);

    ctx.CollectRealTime();

    local_send_msg_time = grape::GetCurrentTime();
    if(ctx.accept()){
      if(ws_scheduler.local_id() == 0){
        LOG(INFO) << "accept";
        ws_scheduler.ShowHint();
      }
      ForEach(stream,
              WorkSourceArray<vertex_t>(out_q_local.data(), out_local_len),
              [=] __device__(vertex_t v) mutable {
                d_dist_curr[v] = min(d_dist_curr[v], d_dist_next[v]);
              });

      ForEach(stream,
              WorkSourceArray<vertex_t>(out_q_remote.data(), out_remote_len),
              [=] __device__(vertex_t v) mutable {
                d_dist_curr[v] = d_dist_next[v];
                d_mm.template SyncStateOnOuterVertex(d_frag, v, d_dist_next[v]);
              });
      stream.Sync();
      in_q.Swap(out_q_local);
    } else {
      LOG(INFO) << "Re run; adjusting";
      ForEach(stream,
              WorkSourceArray<vertex_t>(out_q_local.data(), out_local_len),
              [=] __device__(vertex_t v) mutable {
                d_dist_next[v] = d_dist_curr[v];
              });
      ForEach(stream,
              WorkSourceArray<vertex_t>(out_q_remote.data(), out_remote_len),
              [=] __device__(vertex_t v) mutable {
                d_dist_next[v] = d_dist_curr[v];
              });
      stream.Sync();
      messages.ForceRerun();
    }
    local_send_msg_time =
        (grape::GetCurrentTime() - local_send_msg_time) * 1000;

    std::stringstream ss;
#ifdef PROFILING
    ss << "Stat:" << std::endl;
    ctx.PrintWorkSizeStat(
        ss, in_size,
        ws_scheduler.GetStolenWorkSource(ws_scheduler.local_id()).size(),
        out_local_len, out_remote_len, stolen_work_size, scanned_local_n_edges,
        scanned_remote_n_edges, counter.GetCount(stream),
        counter_ws.GetCount(stream));

    ctx.PrintWorkTimeStat(ss, get_msg_time, ws_scheduling_time,
                          local_compute_time, remote_compute_time,
                          local_send_msg_time, remote_send_msg_time);
#endif
    if (ws_scheduler.local_id() == 0) {
      LOG(INFO) << ss.str();
    }

    ctx.get_msg_time_ += get_msg_time;
    ctx.ws_scheduling_time_ += ws_scheduling_time;
    ctx.remote_compute_time_ += remote_compute_time;
    ctx.remote_send_msg_time_ += remote_send_msg_time;
    ctx.local_compute_time_ += local_compute_time;
    ctx.local_send_msg_time_ += local_send_msg_time;

    if (out_local_len + out_remote_len > 0) {
      messages.ForceContinue();
    }
    messages.RecordUnpackTime(get_msg_time/1000);
    messages.RecordComputeTime(remote_compute_time/1000 + local_compute_time/1000);
    messages.RecordPackTime(local_send_msg_time/1000 + remote_send_msg_time/1000);
  }
};
}  // namespace grape_gpu

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_SEARCH_H_
