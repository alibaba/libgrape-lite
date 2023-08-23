
#ifndef ANALYTICAL_APPS_PHP_PHP_H_
#define ANALYTICAL_APPS_PHP_PHP_H_
#include "grape/grape.h"
namespace grape {
template <typename FRAG_T>
class PHPContext : public ContextBase<FRAG_T> {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

 public:
  using probability_t = double;

  explicit PHPContext() {}

  void Init(const FRAG_T& frag, DefaultMessageManager& messages,
            oid_t source_id, double damping_factor, double php_mr) {
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    this->damping_factor = damping_factor;
    this->curr_round = 0;
    this->php_mr = php_mr;
    weight_sum.Init(inner_vertices, 0);
    result.Init(vertices, 0);
    next_result.Init(vertices, 0);


    native_source = frag.GetInnerVertex(source_id, source);
    this->source_id = source_id;

    if (native_source) {
      result[source] = 1;
      if (frag.GetLocalOutDegree(source) == 0) {
        LOG(FATAL) << "Bad dataset: source vertex has no out-neighbor!";
      }
    }
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto iv = frag.InnerVertices();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << result[v] << std::endl;
    }
  }

  bool native_source;  // source on current fragment;
  typename FRAG_T::oid_t source_id;
  Vertex<vid_t> source;
  double damping_factor;
  int curr_round;
  int php_mr;
  VertexArray<typename FRAG_T::edata_t, vid_t> weight_sum;
  VertexArray<probability_t, vid_t> result;
  VertexArray<probability_t, vid_t> next_result;
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
class PHP : public AppBase<FRAG_T, PHPContext<FRAG_T>>, public Communicator {
 public:
  INSTALL_DEFAULT_WORKER(PHP<FRAG_T>, PHPContext<FRAG_T>, FRAG_T)
  using vertex_t = typename FRAG_T::vertex_t;
  using probability_t = typename context_t::probability_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto iv = frag.InnerVertices();

    for (auto u : iv) {
      auto oes = frag.GetOutgoingAdjList(u);

      for (auto& e : oes) {
        auto v = e.neighbor;
        auto weight = e.data;

        if (frag.GetId(v) != ctx.source_id) {
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

    {
      vertex_t v;
      double rank;
      while (messages.template GetMessage(frag, v, rank)) {
        result[v] += rank;
      }
    }

    if (ctx.curr_round >= ctx.php_mr) {
      return;
    } else {
      messages.ForceContinue();
    }
    ctx.curr_round++;


    for (auto u : iv) {
      auto oes = frag.GetOutgoingAdjList(u);

      for (auto& e : oes) {
        auto v = e.neighbor;
        auto weight = e.data;

        if (frag.GetId(v) != ctx.source_id) {
          next_result[v] +=
              damping_factor * result[u] * weight / ctx.weight_sum[u];
        }
      }
    }

    for (auto u : ov) {
      messages.template SyncStateOnOuterVertex(frag, u, next_result[u]);
    }

    for (auto u : vertices) {
      if (frag.GetId(u) == ctx.source_id) {
        next_result[u] = 1;
      }
      result[u] = 0;
    }

    result.Swap(next_result);
  }
};

}  // namespace grape
#endif  // ANALYTICAL_APPS_PHP_PHP_H_
