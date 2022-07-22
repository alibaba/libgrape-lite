#ifndef EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_FUSED_H_
#define EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_FUSED_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"

namespace grape_gpu {
enum MgState { INITIAL, MIGRATING_SEND, MIGRATING_RECV, MIGRATED };

template <typename FRAG_T>
class SSSPFusedContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using dist_t = uint32_t;

  explicit SSSPFusedContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  void Init(GPUMessageManager& messages, AppConfig app_config, oid_t src_id,
            dist_t init_prio, int sw_round,
            const std::vector<int>& migrate_to) {
    auto& frag = this->fragment();
    auto nv = frag.GetTotalVerticesNum();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto& comm_spec = messages.comm_spec();

    this->src_id = src_id;
    this->lb = app_config.lb;
    this->prio = init_prio;
    this->sw_round = sw_round;
    this->migrate_to = migrate_to;
    this->d_migrate_to = migrate_to;
    // Init Barrier
    {
      CHECK_EQ(migrate_to.size(), comm_spec.local_num())
          << "Illegal param: mg_to";

      int device_rank = -1;
      int num_gpus_required = 0;

      for (int local_id = 0; local_id < comm_spec.local_num(); local_id++) {
        if (migrate_to[local_id] == -1) {
          if (local_id == comm_spec.local_id()) {
            device_rank = num_gpus_required;
          }
          num_gpus_required++;
        }
      }
      LOG(INFO) << "Local id: " << comm_spec.local_id()
                << " dev rank: " << device_rank
                << " num_gpus_required: " << num_gpus_required;
      global_barrier =
          MultiDeviceBarrier(comm_spec, device_rank, num_gpus_required);
    }

    if (migrate_to[comm_spec.local_id()] == -1) {
      work_for.push_back(comm_spec.local_id());

      for (int i = 0; i < comm_spec.local_num(); i++) {
        if (migrate_to[i] == comm_spec.local_id()) {
          work_for.push_back(i);
        }
      }
    }

    values.Init(vertices, std::numeric_limits<dist_t>::max());
    in_q.Init(vertices.size());
    out_q_inner.Init(vertices.size());
    out_q_outer.Init(vertices.size());
    visited.Init(vertices);

    mg_values.resize(nv, std::numeric_limits<dist_t>::max());
    mg_in_q.Init(nv);
    mg_visited.Init(nv);
    mg_out_q.resize(comm_spec.local_num());
    d_mg_out_q.resize(comm_spec.local_num());
    msg_buffer =
        IPCArray<thrust::pair<vid_t, dist_t>, IPCMemoryPlacement::kDevice>(
            comm_spec);
    msg_buffer.Init(nv);
    global_msg_buffer_len =
        IPCArray<uint32_t, IPCMemoryPlacement::kDevice>(comm_spec);
    global_msg_buffer_len.Init(8, 0);
    msg_buffer_last_pos.resize(1, 0);

    for (int local_id = 0; local_id < comm_spec.local_num(); local_id++) {
      mg_out_q[local_id].Init(nv);
      d_mg_out_q[local_id] = mg_out_q[local_id].DeviceObject();
    }

    // Allocate this flag on the first device that involves kernel fusion
    global_active_count = GlobalSharedValue<uint32_t>(
        comm_spec, global_barrier.leader_local_id());
    global_active_count.set(0, messages.stream());
    aggregator =
        AtomicSum<uint32_t>(comm_spec, global_barrier.GetContext().deviceRank,
                            global_barrier.GetContext().numDevices);

    if (init_prio == 0) {
      double weight_sum = 0;

      for (auto v : iv) {
        auto oes = frag.GetOutgoingAdjList(v);
        for (auto& e : oes) {
          weight_sum += e.get_data();
        }
      }
      /**
       * We select a similar heuristic, Δ = cw/d,
          where d is the average degree in the graph, w is the average
          edge weight, and c is the warp width (32 on our GPUs)
          Link: https://people.csail.mit.edu/jshun/papers/DBGO14.pdf
       */
      init_prio = 32 * (weight_sum / frag.GetEdgeNum()) /
                  (1.0 * frag.GetEdgeNum() / iv.size());
    }

    messages.InitBuffer((sizeof(vid_t) + sizeof(dist_t) + 1) * nv,
                        (sizeof(vid_t) + sizeof(dist_t) + 1) * nv);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto vm_ptr = frag.vm_ptr();
    thrust::host_vector<dist_t> h_dist = mg_values;
    std::unordered_set<int> work_for_set(work_for.begin(), work_for.end());

    for (vid_t sid = 0; sid < frag.GetTotalVerticesNum(); sid++) {
      auto gid = frag.ConsecutiveGid2Gid(sid);
      auto fid = vm_ptr->GetFidFromGid(gid);

      if (work_for_set.find(fid) != work_for_set.end()) {
        oid_t oid;

        CHECK(vm_ptr->GetOid(gid, oid));
        os << oid << " " << h_dist[sid] << "\n";
      }
    }
  }

  oid_t src_id{};
  LoadBalancing lb{};
  dist_t prio{};

  bool native_source{};
  int sw_round{};
  int curr_round{};

  // Local stage
  VertexArray<dist_t, vid_t> values;
  Queue<vertex_t, uint32_t> in_q, out_q_inner, out_q_outer;
  DenseVertexSet<vid_t> visited;

  std::vector<int> work_for;
  std::vector<int> migrate_to;  // value represents rank instead of device
  thrust::device_vector<int> d_migrate_to;
  thrust::device_vector<dist_t> mg_values;
  Queue<vid_t, uint32_t> mg_in_q;
  std::vector<Queue<vid_t, uint32_t>> mg_out_q;
  thrust::device_vector<dev::Queue<vid_t, uint32_t>> d_mg_out_q;

  IPCArray<thrust::pair<vid_t, dist_t>, IPCMemoryPlacement::kDevice> msg_buffer;
  IPCArray<uint32_t, IPCMemoryPlacement::kDevice> global_msg_buffer_len;

  AtomicSum<uint32_t> aggregator;

  thrust::device_vector<uint32_t> msg_buffer_last_pos;
  GlobalSharedValue<uint32_t> global_active_count;

  Bitset<vid_t> mg_visited;
  MgState state;
  MultiDeviceBarrier global_barrier;
};

template <typename FRAG_T>
class SSSPFused : public GPUAppBase<FRAG_T, SSSPFusedContext<FRAG_T>>,
                  public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(SSSPFused<FRAG_T>, SSSPFusedContext<FRAG_T>, FRAG_T)
  using dist_t = typename context_t::dist_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto& stream = messages.stream();
    auto src_id = ctx.src_id;
    vertex_t source;

    ctx.native_source = frag.GetInnerVertex(src_id, source);

    if (ctx.native_source) {
      LaunchKernel(
          messages.stream(),
          [=] __device__(dev::VertexArray<dist_t, uint32_t> dist,
                         dev::Queue<vertex_t, uint32_t> in_q) {
            auto tid = TID_1D;

            if (tid == 0) {
              dist[source] = 0;
              in_q.Append(source);
            }
          },
          ctx.values.DeviceObject(), ctx.in_q.DeviceObject());
    }
    messages.ForceContinue();

    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(0);
    messages.RecordPackTime(0);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    auto& stream = messages.stream();
    auto d_frag = frag.DeviceObject();
    auto& in_q = ctx.in_q;
    auto d_in_q = in_q.DeviceObject();
    auto& out_q_inner = ctx.out_q_inner;
    auto d_out_q_inner = out_q_inner.DeviceObject();
    auto& out_q_outer = ctx.out_q_outer;
    auto d_out_q_outer = out_q_outer.DeviceObject();
    auto& visited = ctx.visited;
    auto d_dist = ctx.values.DeviceObject();
    auto d_visited = visited.DeviceObject();

    messages.template ParallelProcess<dev_fragment_t, dist_t>(
        d_frag, [=] __device__(vertex_t v, dist_t received_dist) mutable {
          assert(d_frag.IsInnerVertex(v));

          if (received_dist < atomicMin(&d_dist[v], received_dist)) {
            if (d_visited.Insert(v)) {
              d_in_q.AppendWarp(v);
            }
          }
        });

    // Consume last batch of messages and return
    if (ctx.curr_round == ctx.sw_round) {
      messages.RecordUnpackTime(t_unpack);
      messages.RecordComputeTime(t_compute);
      messages.RecordPackTime(t_pack);
      return;
    }

    out_q_inner.Clear(stream);
    out_q_outer.Clear(stream);
    visited.Clear(stream);

    size_t in_size = in_q.size(stream);

    ForEachOutgoingEdge(
        stream, d_frag, WorkSourceArray<vertex_t>(in_q.data(), in_size),
        [=] __device__(vertex_t u) { return d_dist[u]; },
        [=] __device__(const VertexMetadata<vid_t, dist_t>& metadata,
                       const nbr_t& nbr) mutable {
          dist_t new_depth = metadata.metadata + nbr.get_data();
          vertex_t v = nbr.get_neighbor();
          if (new_depth < atomicMin(&d_dist[v], new_depth)) {
            if (d_visited.Insert(v)) {
              // we use near queue to store active local vertices, far queue
              // to hold remote active vertices
              if (d_frag.IsInnerVertex(v)) {
                d_out_q_inner.Append(v);
              } else {
                d_out_q_outer.Append(v);
              }
            }
          }
        },
        ctx.lb);

    auto local_size = out_q_inner.size(stream);

    VLOG(2) << "fid: " << frag.fid() << " In: " << in_size
            << " Local out: " << local_size
            << " Remote out: " << out_q_outer.size(stream);

    out_q_inner.Swap(in_q);

    messages.MakeOutput(
        stream, frag,
        WorkSourceArray<vertex_t>(out_q_outer.data(), out_q_outer.size(stream)),
        [=] __device__(vertex_t v) mutable {
          return thrust::make_pair(d_frag.GetOuterVertexGid(v), d_dist[v]);
        });
    stream.Sync();

    if (local_size > 0) {
      messages.ForceContinue();
    }
    ctx.curr_round++;

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }

  void MigrateSend(const fragment_t& frag, context_t& ctx,
                   message_manager_t& messages) {
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    auto d_frag = frag.DeviceObject();
    auto d_dist = ctx.values.DeviceObject();
    auto d_visited = ctx.visited.DeviceObject();
    auto iv = d_frag.InnerVertices();
    auto& stream = messages.stream();
    auto& comm_spec = messages.comm_spec();
    auto mg_to = ctx.migrate_to[comm_spec.worker_id()];
    auto d_mm = messages.DeviceObject();

    // start migration
    if (mg_to != -1) {
      // migrate frontier and values
      ForEach(stream, WorkSourceRange<vertex_t>(iv.begin(), iv.size()),
              [=] __device__(vertex_t v) mutable {
                vid_t gid = d_frag.Vertex2Gid(v);
                dist_t dist = d_dist[v];
                char in_frontier = d_visited.Exist(v) ? 1 : 0;

                d_mm.SendToFragmentWarpOpt(
                    mg_to, thrust::make_tuple<vid_t, dist_t, char>(
                               gid, dist, in_frontier));
              });
      stream.Sync();
    }

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }

  void MigrateRecv(const fragment_t& frag, context_t& ctx,
                   message_manager_t& messages) {
    double t_unpack = 0, t_compute = 0, t_pack = 0;

    auto d_frag = frag.DeviceObject();
    auto d_dist = ctx.values.DeviceObject();
    auto d_visited = ctx.visited.DeviceObject();
    auto d_mg_values = ArrayView<dist_t>(ctx.mg_values);
    auto& in_q = ctx.in_q;
    auto iv = d_frag.InnerVertices();
    auto& stream = messages.stream();
    auto& comm_spec = messages.comm_spec();
    auto mg_to = ctx.migrate_to[comm_spec.worker_id()];
    auto d_mm = messages.DeviceObject();

    auto max_val = std::numeric_limits<dist_t>::max();

    if (mg_to == -1) {
      auto d_mg_in_q = ctx.mg_in_q.DeviceObject();
      // copy local state
      ForEach(stream, WorkSourceRange<vertex_t>(iv.begin(), iv.size()),
              [=] __device__(vertex_t v) mutable {
                auto sid = d_frag.Vertex2Sid(v);

                d_mg_values[sid] = d_dist[v];
              });

      ForEach(stream, WorkSourceArray<vertex_t>(in_q.data(), in_q.size(stream)),
              [=] __device__(vertex_t v) mutable {
                assert(d_frag.IsInnerVertex(v));
                d_mg_in_q.AppendWarp(d_frag.GetInnerVertexGid(v));
              });
      // consume and apply messages from owner
      messages.template ParallelProcess<thrust::tuple<vid_t, dist_t, char>>(
          [=] __device__(
              const thrust::tuple<vid_t, dist_t, char>& msg) mutable {
            // Process migrated messages
            auto gid = msg.template get<0>();
            auto dist = msg.template get<1>();
            auto in_frontier = msg.template get<2>();
            auto sid = d_frag.Gid2Sid(gid);

            if (in_frontier == 1) {
              d_mg_in_q.AppendWarp(gid);
            }

            assert(d_mg_values[sid] == max_val);
            d_mg_values[sid] = dist;
          });
      stream.Sync();
    }

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }

  void FuseEval(const fragment_t& frag, context_t& ctx,
                message_manager_t& messages) {
    double t_unpack = 0, t_compute = 0, t_pack = 0;
    auto d_frags = frag.DeviceObjects();
    auto d_local_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    fid_t fid = frag.fid(), fnum = frag.fnum();
    auto& mg_values = ctx.mg_values;
    auto d_mg_values = ArrayView<dist_t>(mg_values);
    auto& comm_spec = messages.comm_spec();
    auto& migrate_to = ctx.migrate_to;
    ArrayView<int> d_migrate_to(ctx.d_migrate_to);
    auto d_visited = ctx.mg_visited.DeviceObject();
    auto d_in_q = ctx.mg_in_q.DeviceObject();
    auto d_out_q = ArrayView<dev::Queue<vid_t, uint32_t>>(ctx.d_mg_out_q);

    auto d_msg_buffer = ctx.msg_buffer.device_views();
    auto d_global_msg_buffer_len = ctx.global_msg_buffer_len.device_views();
    auto* d_msg_buffer_last_pos =
        thrust::raw_pointer_cast(ctx.msg_buffer_last_pos.data());
    auto* d_global_active_count = ctx.global_active_count.data();

    LaunchKernel(stream, [=] __device__() mutable {
      if (TID_1D == 0) {
        atomicAdd_system(d_global_active_count, d_in_q.size());
      }
    });

    stream.Sync();

    MPI_Barrier(comm_spec.local_comm());

    LOG(INFO) << "fid: " << fid << " In: " << ctx.mg_in_q.size(stream);

    if (ctx.global_barrier.leader_local_id() == comm_spec.local_id()) {
      LOG(INFO) << "Global In count: " << ctx.global_active_count.get(stream);
    }
    auto barrier_ctx = ctx.global_barrier.GetContext();

    // This is a worker
    if (migrate_to[comm_spec.worker_id()] == -1) {
      auto begin = grape::GetCurrentTime();
      auto d_aggregator = ctx.aggregator.DeviceObject();

      LaunchCooperativeKernel(stream, [=] __device__() mutable {
        auto grid = cooperative_groups::this_grid();
        PeerGroup pg(barrier_ctx, grid);
        // global
        volatile uint32_t& global_active_count = *d_global_active_count;
        // local
        // N.B. out_qs is a local variable
        dev::Queue<vid_t, uint32_t> out_qs[8];
        uint32_t prefix_sum[8][9];

        for (fid_t dst_fid = 0; dst_fid < fnum; dst_fid++) {
          out_qs[dst_fid] = d_out_q[dst_fid];
          for (int i = 0; i < 9; i++) {
            prefix_sum[dst_fid][i] = 0;
          }
        }

        while (global_active_count > 0) {
          // Get Messages
          uint32_t msg_len = prefix_sum[fid][fnum];
          auto msg_buffer = d_msg_buffer[fid];

          for (auto msg_idx = TID_1D; msg_idx < msg_len;
               msg_idx += TOTAL_THREADS_1D) {
            auto& msg = msg_buffer[msg_idx];
            auto gid = msg.first;
            auto dist = msg.second;
            auto sid = d_local_frag.Gid2Sid(gid);

            if (dist < atomicMin(&d_mg_values[sid], dist)) {
              //              if (d_visited.set_bit_atomic(sid)) {
              d_in_q.Append(gid);
              //              }
            }
          }
          // d_visited.clear();
          // Global barrier to make sure messages are consumed before next
          // writing
          pg.sync();

          for (auto i = TID_1D; i < d_in_q.size(); i += TOTAL_THREADS_1D) {
            auto u_gid = d_in_q[i];
            auto u_sid = d_local_frag.Gid2Sid(u_gid);
            auto u_fid = d_local_frag.Gid2Fid(u_gid);
            auto& d_frag = d_frags[u_fid];
            vertex_t u;
            bool ok = d_frag.Gid2Vertex(u_gid, u);
            auto dist = d_mg_values[u_sid];
            assert(ok);

            for (auto e : d_frag.GetOutgoingAdjList(u)) {
              auto v = e.get_neighbor();
              auto v_gid = d_frag.Vertex2Gid(v);
              auto v_sid = d_frag.Gid2Sid(v_gid);
              auto v_fid = d_frag.GetFragId(v_gid);
              auto new_dist = dist + e.get_data();

              if (new_dist < atomicMin(&d_mg_values[v_sid], new_dist)) {
                //                if (d_visited.set_bit_atomic(v_sid)) {
                out_qs[v_fid].Append(v_gid);
                //                }
              }
            }
          }

          // Send messages
          pg.sync();

          uint32_t total_out_size = 0;

          if (pg.thread_rank() == 0) {
            for (fid_t i = 0; i < fnum; i++) {
              for (fid_t j = 0; j < fnum; j++) {
                d_global_msg_buffer_len[i][j] = 0;
              }
            }
          }

          pg.sync();

          for (fid_t i = 0; i < fnum; i++) {
            auto dst_fid = (fid + i) % fnum;
            auto& d_out_q = out_qs[dst_fid];
            auto len = d_out_q.size();

            total_out_size += len;
            if (dst_fid == fid) {
              d_in_q.Swap(d_out_q);
            } else {
              auto redir_dst_fid =
                  d_migrate_to[dst_fid] == -1 ? dst_fid : d_migrate_to[dst_fid];

              if (redir_dst_fid == fid) {
                for (auto idx = TID_1D; idx < len; idx += TOTAL_THREADS_1D) {
                  auto gid = d_out_q[idx];
                  d_in_q.Append(gid);
                }
              } else {
                if (TID_1D == 0) {
                  // += is fine since there's no race condition
                  d_global_msg_buffer_len[fid][dst_fid] += len;
                }
              }
            }
            grid.sync();
          }

          pg.sync();

          // calculate prefix sum arrays
          for (fid_t dst_fid = 0; dst_fid < fnum; dst_fid++) {
            int i = 1;

            prefix_sum[dst_fid][0] = 0;
            for (fid_t src_fid = 0; src_fid < fnum; src_fid++) {
              prefix_sum[dst_fid][i] =
                  prefix_sum[dst_fid][i - 1] +
                  d_global_msg_buffer_len[src_fid][dst_fid];
              i++;
            }
          }

          for (fid_t i = 0; i < fnum; i++) {
            auto dst_fid = (fid + i) % fnum;
            auto& d_out_q = out_qs[dst_fid];

            if (dst_fid == fid) {
              ;
            } else {
              auto redir_dst_fid =
                  d_migrate_to[dst_fid] == -1 ? dst_fid : d_migrate_to[dst_fid];

              if (redir_dst_fid == fid) {
                ;
              } else {
                auto last_pos = prefix_sum[dst_fid][fid];
                auto len =
                    prefix_sum[dst_fid][fid + 1] - prefix_sum[dst_fid][fid];

                for (auto idx = TID_1D; idx < len; idx += TOTAL_THREADS_1D) {
                  auto gid = d_out_q[idx];
                  auto sid = d_local_frag.Gid2Sid(gid);

                  // write to remote buffer with IPC
                  d_msg_buffer[redir_dst_fid][last_pos + idx] =
                      thrust::make_pair(gid, d_mg_values[sid]);
                }
              }
            }
            grid.sync();
            d_out_q.Clear();
          }

          // Global barrier is needed to ensure that the messages are written to
          // remote buffer before reading in the next round
          global_active_count = 0;  // reset counter, benign race condition
          pg.sync();
          d_aggregator.sum(total_out_size, &global_active_count, pg);
          pg.sync();
        }
      });
      stream.Sync();

      LOG(INFO) << "Time: " << grape::GetCurrentTime() - begin;
    }

    messages.RecordUnpackTime(t_unpack);
    messages.RecordComputeTime(t_compute);
    messages.RecordPackTime(t_pack);
  }
};
}  // namespace grape_gpu

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_FUSED_H_
