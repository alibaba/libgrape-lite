#ifndef AUTOINC_EXAMPLES_ANALYTICAL_APPS_D2UD_H_
#define AUTOINC_EXAMPLES_ANALYTICAL_APPS_D2UD_H_
#include "grape/app/app_base.h"
#include "grape/grape.h"

namespace grape {
template <typename FRAG_T>
class D2UDContext : public ContextBase<FRAG_T> {
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertext_t = Vertex<vid_t>;
  using edata_t = typename FRAG_T::edata_t;
  bool weighted_{};

 public:
  explicit D2UDContext() = default;

  void Init(const FRAG_T& frag, DefaultMessageManager& messages,
            bool weighted) {
    weighted_ = weighted;
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto iv = frag.InnerVertices();

    for (auto u : iv) {
      std::map<vertext_t, edata_t> dsts;
      auto oes = frag.GetOutgoingAdjList(u);

      for (auto& e : oes) {
        auto weight = e.data;
        auto v = e.neighbor;

        dsts.template emplace(v, weight);
      }

      for (auto& e : dsts) {
        if (weighted_) {
          os << frag.GetId(u) << " " << frag.GetId(e.first) << " " << e.second
             << std::endl;
        } else {
          os << frag.GetId(u) << " " << frag.GetId(e.first) << std::endl;
        }
      }
    }
  }
};

template <typename FRAG_T>
class D2UD : public AppBase<FRAG_T, D2UDContext<FRAG_T>> {
  using vertex_t = typename FRAG_T::vertex_t;
  INSTALL_DEFAULT_WORKER(D2UD<FRAG_T>, D2UDContext<FRAG_T>, FRAG_T);
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {}

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {}
};

}  // namespace grape
#endif  // AUTOINC_EXAMPLES_ANALYTICAL_APPS_D2UD_H_
