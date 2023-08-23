
#ifndef LIBGRAPE_LITE_GRAPE_APP_TRAVERSAL_APP_BASE_H_
#define LIBGRAPE_LITE_GRAPE_APP_TRAVERSAL_APP_BASE_H_

#include "grape/types.h"
#include "grape/utils/dependency_data.h"
#include "grape/utils/vertex_array.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class TraversalAppBase {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using value_t = VALUE_T;
  using delta_t = DependencyData<vid_t, value_t>;
  using vertex_t = typename fragment_t::vertex_t;
  static constexpr bool need_split_edges = false;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;

  TraversalAppBase() = default;

  virtual ~TraversalAppBase() = default;

  virtual void IterateBegin() {}

  virtual value_t GetInitValue(const vertex_t& v) const = 0;

  virtual delta_t GetInitDelta(const vertex_t& v) const = 0;

  virtual bool CombineValueDelta(value_t& lhr, const delta_t& rhs) = 0;

  /**
   * Atomic required
   * @param v
   * @param val
   * @return
   */
  // virtual bool AccumulateDelta(delta_t& lhs, const delta_t& rhs) = 0;

  // bool AccumulateTo(const vertex_t& v, const delta_t& delta) {
  //   return AccumulateDelta(deltas_[v], delta);
  // }
virtual bool accumulate(delta_t& lhs, const delta_t& rhs) = 0;

  bool AccumulateTo(const vertex_t& v, const delta_t& delta) {
    return accumulate(deltas_[v], delta);
  }
  virtual value_t GetPriority(const vertex_t& v, const value_t& value,
                              const delta_t& delta) const {
    return GetIdentityElement();
  }

  virtual void Compute(const vertex_t& u, const value_t& value,
                       const delta_t& delta,
                       DenseVertexSet<vid_t>& modified) = 0;

  virtual value_t GetIdentityElement() const = 0;

  inline vid_t DeltaParentGid(vertex_t v) { return deltas_[v].parent_gid; }

  delta_t GenDelta(vertex_t parent, value_t val) const {
    return {frag_->Vertex2Gid(parent), val};
  }

  std::vector<value_t> DumpResult() {
    std::vector<value_t> result;

    for (auto v : values_.GetVertexRange()) {
      result.push_back(values_[v]);
    }
    return result;
  }

  const fragment_t& fragment() const { return *frag_; }

 protected:
  void Init(const CommSpec& comm_spec, std::shared_ptr<FRAG_T>& frag) {
    auto vertices = frag->Vertices();
    auto inner_vertices = frag->InnerVertices();

    frag_ = frag;
    values_.Init(inner_vertices);
    deltas_.Init(vertices);

    curr_modified_.Init(vertices);
    next_modified_.Init(vertices);

    for (auto v : inner_vertices) {
      values_[v] = GetInitValue(v);
    }

    for (auto v : vertices) {
      deltas_[v] = GetInitDelta(v);

      if (deltas_[v].value != GetIdentityElement()) {
        curr_modified_.Insert(v);
      }
    }

    uint64_t memory = 0, global_mem;
    memory += curr_modified_.Range().size() / 64;
    memory += next_modified_.Range().size() / 64;
    memory += sizeof(value_t) * values_.size();
    memory += sizeof(delta_t) * deltas_.size();

    Communicator communicator;
    communicator.InitCommunicator(comm_spec.comm());
    communicator.template Sum(memory, global_mem);

    if (batch_stage_) {
      batch_stage_ = false;
      if (comm_spec.worker_id() == grape::kCoordinatorRank) {
        LOG(INFO) << "Mem: " << global_mem / 1024 / 1024 << " MB";
      }
    }
  }

  std::shared_ptr<fragment_t> frag_;
  DenseVertexSet<vid_t> curr_modified_, next_modified_;
  VertexArray<value_t, vid_t> values_{};
  VertexArray<delta_t, vid_t> deltas_{};
  bool batch_stage_{true};
  template <typename APP_T>
  friend class IngressSyncTraversalWorker;
};

}  // namespace grape
#endif  // LIBGRAPE_LITE_GRAPE_APP_TRAVERSAL_APP_BASE_H_
