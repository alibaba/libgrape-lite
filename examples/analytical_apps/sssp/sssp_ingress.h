
#ifndef ANALYTICAL_APPS_SSSP_SSSP_INGRESS_H_
#define ANALYTICAL_APPS_SSSP_SSSP_INGRESS_H_

#include "flags.h"
#include "grape/app/traversal_app_base.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class SSSPIngress : public TraversalAppBase<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = typename TraversalAppBase<FRAG_T, VALUE_T>::value_t;
  using delta_t = typename TraversalAppBase<FRAG_T, VALUE_T>::delta_t;

  value_t GetInitValue(const vertex_t& v) const override {
    return GetIdentityElement();
  }

  delta_t GetInitDelta(const vertex_t& v) const override {
    vertex_t source;
    bool native_source =
        this->fragment().GetInnerVertex(FLAGS_sssp_source, source);
    value_t init_dist = GetIdentityElement();

    if (native_source && source == v) {
      init_dist = 0;
    }

    return this->GenDelta(v, init_dist);
  }

  bool CombineValueDelta(value_t& lhs, const delta_t& rhs) override {
    if (lhs > rhs.value) {
      lhs = rhs.value;
      return true;
    }
    return false;
  }

  // bool AccumulateDelta(delta_t& lhs, const delta_t& rhs) override {
  //   return lhs.SetIfLessThan(rhs);
  // }

  bool accumulate(delta_t& lhs, const delta_t& rhs) override {
    return atomic_min(lhs.value , rhs.value);
  }
  value_t GetPriority(const vertex_t& v, const value_t& value,
                      const delta_t& delta) const override {
    return delta.value;
  }
  value_t generate(value_t v, value_t m, value_t w) {
    return v+m;
  }
  void Compute(const vertex_t& u, const value_t& value, const delta_t& delta,
               DenseVertexSet<vid_t>& modified) override {
    auto dist = delta.value;
    auto oes = this->fragment().GetOutgoingAdjList(u);

    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        // auto new_dist = e.data + dist;
        auto new_dist = generate(0, e.data , dist);
        delta_t delta_to_send = this->GenDelta(u, new_dist);

        if (this->AccumulateTo(v, delta_to_send)) {
          modified.Insert(v);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        // auto new_dist = e.data + dist;
        auto new_dist = generate(0, e.data,dist);
        delta_t delta_to_send = this->GenDelta(u, new_dist);

        if (this->AccumulateTo(v, delta_to_send)) {
          modified.Insert(v);
        }
      }
    }
  }

  value_t GetIdentityElement() const override {
    return std::numeric_limits<value_t>::max();
  }
};

}  // namespace grape

#endif  // ANALYTICAL_APPS_SSSP_SSSP_INGRESS_H_
