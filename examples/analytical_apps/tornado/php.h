#ifndef EXAMPLES_ANALYTICAL_APPS_TORNADO_PHP_H_
#define EXAMPLES_ANALYTICAL_APPS_TORNADO_PHP_H_
#include <iomanip>

#include "grape/app/vertex_data_context.h"
#include "grape/grape.h"
#include "grape/worker/tornado_worker.h"

namespace grape {

namespace tornado {

template <typename FRAG_T>
class PHPContext : public grape::VertexDataContext<FRAG_T, double> {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

 public:
  using probability_t =
      typename grape::VertexDataContext<FRAG_T, double>::data_t;

  explicit PHPContext()
      : VertexDataContext<FRAG_T, double>(true), result(this->data()) {}

  void Init(const FRAG_T& frag, ParallelMessageManager& messages,
            const CommSpec& comm_spec, oid_t source_id, double damping_factor,
            double php_tol) {
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    this->damping_factor = damping_factor;
    this->php_tol = php_tol;
    weight_sum.Init(inner_vertices, 0);
    last_php.Init(inner_vertices, 0);
    result.Init(vertices, 0);

    CHECK(frag.Oid2Gid(source_id, source_gid));
    Vertex<vid_t> source;
    bool native_source = frag.GetInnerVertex(source_id, source);

    if (native_source) {
      result[source] = 1;
      if (frag.GetLocalOutDegree(source) == 0) {
        LOG(FATAL) << "Bad dataset: source vertex has no out-neighbor!";
      }
    }

    uint64_t memory = 0, global_mem;

    memory += 2 * sizeof(probability_t) * result.GetVertexRange().size();

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

  typename FRAG_T::vid_t source_gid;
  double damping_factor;
  VertexArray<uint64_t, vid_t> weight_sum;
  double php_tol;
  VertexArray<probability_t, vid_t>& result;
  VertexArray<probability_t, vid_t> next_result;
  VertexArray<probability_t, vid_t> last_php;
};

/**
 * PHP is used to measure the proximity (similarity) between a given source
 * vertex s and any other vertex j. As a random walk based algorithm, a walker
 * at vertex i moves to i’s out-neighbor j with a probability proportional to an
 * edge weight w(i, j). The sum of transition probabilities indicates the
 * proximity. In particular, s as the query vertex has a constant proximity
 * value 1.
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PHP : public ParallelAppBase<FRAG_T, PHPContext<FRAG_T>>,
            public ParallelEngine,
            public Communicator {
 public:
  using fragment_t = FRAG_T;
  using context_t = PHPContext<FRAG_T>;
  using message_manager_t = ParallelMessageManager;
  using app_t = PHP<fragment_t>;
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

    ctx.next_result.Init(vertices, 0);
    messages.InitChannels(thread_num());

    for (auto u : iv) {
      auto oes = frag.GetOutgoingAdjList(u);

      for (auto& e : oes) {
        auto v = e.neighbor;
        auto weight = e.data;

        if (frag.Vertex2Gid(v) != ctx.source_gid) {
          ctx.weight_sum[u] += weight;
        }
      }
    }

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto& result = ctx.result;
    auto& next_result = ctx.next_result;
    auto damping_factor = ctx.damping_factor;

    messages.template ParallelProcess<fragment_t, double>(
        thread_num(), frag, [&result](int tid, vertex_t v, double rank) {
          atomic_add(result[v], rank);
        });

    double local_diff_sum = 0, global_diff_sum;

    for (auto v : iv) {
      local_diff_sum += fabs(ctx.last_php[v] - result[v]);
      ctx.last_php[v] = result[v];
    }
    Sum(local_diff_sum, global_diff_sum);

    if (frag.fid() == 0) {
      LOG(INFO) << "Diff: " << global_diff_sum;
    }

    if (global_diff_sum > 0 && global_diff_sum < ctx.php_tol) {
      return;
    } else {
      messages.ForceContinue();
    }

    ForEach(iv, [&frag, &ctx, &result, &next_result, damping_factor](
                    int tid, vertex_t u) {
      auto oes = frag.GetOutgoingAdjList(u);

      for (auto& e : oes) {
        auto v = e.neighbor;
        auto weight = 1.0 * e.data / ctx.weight_sum[u];

        if (frag.Vertex2Gid(v) != ctx.source_gid) {
          atomic_add(next_result[v], damping_factor * result[u] * weight);
        }
      }
    });

    ForEach(ov, [&messages, &frag, &next_result](int tid, vertex_t u) {
      messages.Channels()[tid].SyncStateOnOuterVertex(frag, u, next_result[u]);
    });

    ForEach(vertices,
            [&frag, &ctx, &result, &next_result](int tid, vertex_t u) {
              if (frag.Vertex2Gid(u) == ctx.source_gid) {
                next_result[u] = 1;
              }
              result[u] = 0;
            });

    result.Swap(next_result);
  }
};

}  // namespace tornado

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_TORNADO_PHP_H_
