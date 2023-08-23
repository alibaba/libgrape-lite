
#ifndef EXAMPLES_ANALYTICAL_APPS_TORNADO_PAGERANK_H_
#define EXAMPLES_ANALYTICAL_APPS_TORNADO_PAGERANK_H_
#include <iomanip>

#include "grape/app/vertex_data_context.h"
#include "grape/grape.h"
#include "grape/worker/tornado_worker.h"

namespace grape {

namespace tornado {

template <typename FRAG_T>
class PageRankContext : public grape::VertexDataContext<FRAG_T, float> {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

 public:
  using rank_t = typename grape::VertexDataContext<FRAG_T, float>::data_t;

  explicit PageRankContext()
      : VertexDataContext<FRAG_T, float>(true), result(this->data()) {}

  void Init(const FRAG_T& frag, ParallelMessageManager& messages,
            const CommSpec& comm_spec, float damping_factor, rank_t pr_tol) {
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    this->damping_factor = damping_factor;
    this->pr_tol = pr_tol;
    last_pr.Init(inner_vertices);

    for (auto v : vertices) {
      result[v] = 0;
    }

    uint64_t memory = 0, global_mem;

    memory += 2 * sizeof(rank_t) * result.GetVertexRange().size();

    Communicator communicator;
    communicator.InitCommunicator(comm_spec.comm());
    communicator.template Sum(memory, global_mem);
    if (comm_spec.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Mem: " << global_mem / 1024 / 1024 << " MB";
    }
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto iv = frag.InnerVertices();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << result[v] << std::endl;
    }
  }

  float damping_factor;
  rank_t pr_tol;
  rank_t dangling_sum;
  VertexArray<rank_t, vid_t>& result;
  VertexArray<rank_t, vid_t> next_result;
  VertexArray<rank_t, vid_t> last_pr;
};

template <typename FRAG_T>
class PageRank : public ParallelAppBase<FRAG_T, PageRankContext<FRAG_T>>,
                 public ParallelEngine,
                 public Communicator {
 public:
  using fragment_t = FRAG_T;
  using context_t = PageRankContext<FRAG_T>;
  using message_manager_t = ParallelMessageManager;
  using app_t = PageRank<fragment_t>;
  using worker_t = TornadoWorker<app_t>;
  static std::shared_ptr<worker_t> CreateWorker(std::shared_ptr<app_t> app) {
    return std::shared_ptr<worker_t>(new worker_t(app));
  }

  using vertex_t = typename FRAG_T::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto iv = frag.InnerVertices();
    auto vertices = frag.Vertices();

    messages.InitChannels(thread_num());
    ctx.next_result.Init(vertices, 0);
    for (auto v : iv) {
      ctx.last_pr[v] = 0;
    }

    ctx.dangling_sum = 0;

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto& result = ctx.result;
    auto& next_result = ctx.next_result;
    auto damping_factor = ctx.damping_factor;
    auto& dangling_sum = ctx.dangling_sum;

    messages.template ParallelProcess<fragment_t, float>(
        thread_num(), frag, [&result](int tid, vertex_t v, float rank) {
          atomic_add(result[v], rank);
        });

    float local_diff_sum = 0, global_diff_sum;

    for (auto v : iv) {
      local_diff_sum += fabs(ctx.last_pr[v] - result[v]);
      ctx.last_pr[v] = result[v];
    }
    Sum(local_diff_sum, global_diff_sum);

    if (frag.fid() == 0) {
      LOG(INFO) << "Diff: " << global_diff_sum;
    }

    if (global_diff_sum > 0 && global_diff_sum < ctx.pr_tol) {
      return;
    } else {
      messages.ForceContinue();
    }

    float next_dangling_sum = 0;

    ForEach(iv, [&frag, &next_result, damping_factor, dangling_sum](
                    int tid, vertex_t u) {
      next_result[u] =
          (1 - damping_factor) / frag.GetTotalVerticesNum() +
          damping_factor * dangling_sum / frag.GetTotalVerticesNum();
    });

    for (auto v : iv) {
      if (frag.GetLocalOutDegree(v) == 0) {
        next_dangling_sum += result[v];
      }
    }

    ForEach(iv, [&ctx, &frag, &result, &next_result, damping_factor](
                    int tid, vertex_t u) {
      auto oes = frag.GetOutgoingAdjList(u);

      for (auto& e : oes) {
        auto v = e.neighbor;

        atomic_add(next_result[v],
                   damping_factor * result[u] / frag.GetLocalOutDegree(u));
      }
      result[u] = 0;
    });

    ForEach(ov, [&frag, &messages, &next_result](int tid, vertex_t v) {
      messages.Channels()[tid].SyncStateOnOuterVertex(frag, v, next_result[v]);
      next_result[v] = 0;
    });

    Sum(next_dangling_sum, dangling_sum);
    result.Swap(next_result);
  }
};

}  // namespace tornado

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_TORNADO_PAGERANK_H_
