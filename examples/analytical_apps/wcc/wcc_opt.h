#ifndef EXAMPLES_ANALYTICAL_APPS_WCC_WCC_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_WCC_WCC_OPT_H_
#include "app_config.h"
#include "grape_gpu/grape_gpu.h"
DEFINE_int32(wcc_batch_size, 0, "");
DEFINE_int32(wcc_non_atomic, 0, "");

namespace grape_gpu {
template <typename FRAG_T>
class WCCOptContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;

  explicit WCCOptContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

  ~WCCOptContext() {
    auto& frag = this->fragment();

    if (frag.fid() == 0) {
      thrust::host_vector<vid_t> h_parents(parents);
      std::unordered_set<vid_t> cc_ids;

      cc_ids.template insert(h_parents.begin(), h_parents.end());

      LOG(INFO) << "CC Count: " << cc_ids.size();
    }
  }

  void Init(GPUMessageManager& messages, AppConfig app_config) {
    auto& frag = this->fragment();

    // After being converted to coo, the internal csr/csc is released
    coo_frag =
        const_cast<FRAG_T&>(frag).ConvertToCOO(COOIdType::kConsecutive, true);
    thrust::host_vector<vid_t> h_parents;
    // Init parent id

    h_parents.resize(frag.GetTotalVerticesNum());

    for (vid_t gid = 0; gid < frag.GetTotalVerticesNum(); gid++) {
      h_parents[gid] = gid;
    }
    parents = h_parents;

    if (frag.fid() == 0) {
      messages.InitBuffer(0, frag.GetTotalVerticesNum() * sizeof(vid_t) * 2);
    } else {
      messages.InitBuffer(frag.Vertices().size() * sizeof(vid_t) * 2, 0);
    }
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();

    if (frag.fid() == 0) {
      thrust::host_vector<vid_t> h_parents = parents;

      std::unordered_set<vid_t> cc_ids;

      for (vid_t offset = 0; offset < frag.GetTotalVerticesNum(); offset++) {
        auto gid = coo_frag->ConsecutiveGid2Gid(offset);
        auto p_gid = coo_frag->ConsecutiveGid2Gid(h_parents[offset]);

        os << frag.Gid2Oid(gid) << " " << frag.Gid2Oid(p_gid) << std::endl;
        cc_ids.insert(h_parents[offset]);
      }
      LOG(INFO) << "CC Count: " << cc_ids.size();
    }
  }

  LoadBalancing lb{};
  thrust::device_vector<vid_t> parents;
  std::shared_ptr<typename FRAG_T::coo_t> coo_frag;
  Counter counter_;
};

template <typename FRAG_T>
class WCCOpt : public GPUAppBase<FRAG_T, WCCOptContext<FRAG_T>>,
               public Communicator,
               public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(WCCOpt<FRAG_T>, WCCOptContext<FRAG_T>, FRAG_T)
  using dev_fragment_t = typename fragment_t::device_t;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto& coo = *ctx.coo_frag;
    auto d_coo = coo.DeviceObject();
    auto iv = coo.InnerVerticesWithConsecutiveGid();
    auto dev_mm = messages.DeviceObject();
    auto* d_parents = thrust::raw_pointer_cast(ctx.parents.data());
    auto& stream = messages.stream();
    double compute = 0, pack = 0;
    size_t n_iter = 0;

    compute -= grape::GetCurrentTime();

    // HookHighToLowAtomic
    size_t begin_eid = 0;
    size_t batch_size = fLI::FLAGS_wcc_batch_size;

    if (batch_size == 0) {
      batch_size = iv.size();
    }

    double hook = 0, jump = 0;

    auto& counter = ctx.counter_;

    counter.ResetAsync(stream);
    auto d_counter = counter.DeviceObject();

    auto multi_jump = [&](const Stream& s) {
      // MultiJumpCompress
      ForEach(s, coo.Vertices(), [=] __device__(vertex_t v) mutable {
        auto v_gid = v.GetValue();
        auto p = d_parents[v_gid];
        auto pp = d_parents[p];

        while (p != pp) {
          d_parents[v_gid] = pp;
          p = pp;
          pp = d_parents[p];
        }
      });
    };

    for (int non_atomic = 0; non_atomic < fLI::FLAGS_wcc_non_atomic;
         non_atomic++) {
      while (begin_eid < d_coo.GetEdgeNum()) {
        size_t end_eid = std::min(begin_eid + batch_size, d_coo.GetEdgeNum());

        hook -= grape::GetCurrentTime();
        LaunchKernel(stream, batch_size, [=] __device__() mutable {
          auto tid = TID_1D;
          auto nthreads = TOTAL_THREADS_1D;

          for (size_t i = begin_eid + tid; i < end_eid; i += nthreads) {
            auto& e = d_coo[i];
            auto u = e.src();
            auto v = e.dst();

            // is not a self-cycle
            if (u != v) {
              auto p_u = d_parents[u.GetValue()];
              auto p_v = d_parents[v.GetValue()];

              if (p_u != p_v) {
                auto high = p_u > p_v ? p_u : p_v;
                auto low = p_u + p_v - high;

                d_parents[high] = low;
              }
            }
          }
        });
        stream.Sync();
        hook += grape::GetCurrentTime();

        jump -= grape::GetCurrentTime();
        multi_jump(stream);
        stream.Sync();
        jump += grape::GetCurrentTime();
        begin_eid = end_eid;
        n_iter++;
      }
    }

    LOG(INFO) << frag.fid() << " Hook: " << hook * 1000
              << " Jump: " << jump * 1000 << " n_iter: " << n_iter;

    hook = 0;
    jump = 0;
    n_iter = 0;

    begin_eid = 0;

    while (begin_eid < d_coo.GetEdgeNum()) {
      size_t end_eid = std::min(begin_eid + batch_size, d_coo.GetEdgeNum());

      hook -= grape::GetCurrentTime();
      LaunchKernel(stream, batch_size, [=] __device__() mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (size_t i = begin_eid + tid; i < end_eid; i += nthreads) {
          auto& e = d_coo[i];
          auto u = e.src();
          auto v = e.dst();

          // is not a self-cycle
          if (u != v) {
            auto p_u = d_parents[u.GetValue()];
            auto p_v = d_parents[v.GetValue()];

            while (p_u != p_v) {
              auto high = p_u > p_v ? p_u : p_v;
              auto low = p_u + p_v - high;
              auto prev = atomicCAS(&d_parents[high], high, low);

              // Update successfully or parent is updated with low
              if (prev == high || prev == low) {
                break;
              }
              p_u = d_parents[prev];
              p_v = d_parents[low];
              //              d_counter.Add();
            }
          }
        }
      });
      stream.Sync();
      hook += grape::GetCurrentTime();

      jump -= grape::GetCurrentTime();
      multi_jump(stream);
      stream.Sync();
      jump += grape::GetCurrentTime();
      begin_eid = end_eid;
      n_iter++;
    }
    stream.Sync();
    compute += grape::GetCurrentTime();

    LOG(INFO) << frag.fid() << " Hook: " << hook * 1000
              << " Jump: " << jump * 1000 << " n_iter: " << n_iter
              << " Failed count: " << counter.GetCount(stream);

    pack -= grape::GetCurrentTime();
    if (frag.fid() > 0) {
      ForEach(stream, coo.Vertices(), [=] __device__(vertex_t v) mutable {
        auto v_oid = v.GetValue();
        auto parent = d_parents[v_oid];

        if (v_oid != parent) {
          dev_mm.template SendToFragment(0, thrust::make_pair(v_oid, parent));
        }
      });
    }
    stream.Sync();
    pack += grape::GetCurrentTime();

    messages.RecordUnpackTime(0);
    messages.RecordComputeTime(compute);
    messages.RecordPackTime(pack);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto* d_parents = thrust::raw_pointer_cast(ctx.parents.data());
    auto& stream = messages.stream();
    auto& coo = *ctx.coo_frag;
    auto iv = coo.InnerVerticesWithConsecutiveGid();
    double unpack = 0, compute = 0;
    auto& counter = ctx.counter_;

    counter.ResetAsync(stream);
    auto d_counter = counter.DeviceObject();

    auto hook_high_to_low = [=] __device__(vid_t v_gid, vid_t parent) mutable {
      if (v_gid != parent) {
        auto p_u = d_parents[v_gid];
        auto p_v = d_parents[parent];

        while (p_u != p_v) {
          auto high = p_u > p_v ? p_u : p_v;
          auto low = p_u + p_v - high;
          auto prev = atomicCAS(&d_parents[high], high, low);

          if (prev == high || prev == low) {
            break;
          }

          p_u = d_parents[prev];
          p_v = d_parents[low];
          //          d_counter.Add();
        }
      }
    };

    unpack -= grape::GetCurrentTime();
    // HookHighToLowAtomic
    messages.template ParallelProcess<thrust::pair<vid_t, vid_t>>(
        [=] __device__(const thrust::pair<vid_t, vid_t>& msg) mutable {
          auto offset = msg.first;
          auto parent = msg.second;

          hook_high_to_low(offset, parent);
        });
    unpack += grape::GetCurrentTime();

    if (frag.fid() == 0) {
      LOG(INFO) << "Stage2 failed count: " << counter.GetCount(stream);
    }

    compute -= grape::GetCurrentTime();
    ForEach(stream, WorkSourceRange<vertex_t>(iv.begin(), iv.size()),
            [=] __device__(vertex_t v) mutable {
              auto v_oid = v.GetValue();
              auto parent = d_parents[v_oid];

              hook_high_to_low(v_oid, parent);
            });

    if (frag.fnum() > 1 && frag.fid() == 0) {
      size_t graph_vnum = frag.GetTotalVerticesNum();
      // MultiJumpCompress
      LaunchKernel(stream, [=] __device__() mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (auto v_offset = tid; v_offset < graph_vnum; v_offset += nthreads) {
          auto p = d_parents[v_offset];
          auto pp = d_parents[p];

          while (p != pp) {
            d_parents[v_offset] = pp;
            p = pp;
            pp = d_parents[p];
          }
        }
      });
    }
    stream.Sync();
    compute += grape::GetCurrentTime();

    messages.RecordUnpackTime(unpack);
    messages.RecordComputeTime(compute);
    messages.RecordPackTime(0);
  }
};
}  // namespace grape_gpu
#endif  // EXAMPLES_ANALYTICAL_APPS_WCC_WCC_OPT_H_
