
#ifndef LIBGRAPE_LITE_GRAPE_APP_INGRESS_APP_BASE_H_
#define LIBGRAPE_LITE_GRAPE_APP_INGRESS_APP_BASE_H_

#include "grape/types.h"
#include "grape/utils/vertex_array.h"

namespace grape {
template <typename FRAG_T, typename VALUE_T>
class IterateKernel {
 public:
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using value_t = VALUE_T;
  using vertex_t = typename fragment_t::vertex_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  static constexpr bool need_split_edges = false;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;

  explicit IterateKernel() = default;

  virtual ~IterateKernel() = default;

  virtual void init_c(const vertex_t v, value_t& delta, const FRAG_T& frag) {}

  virtual void init_c(const FRAG_T& frag, const vertex_t v, value_t& delta,
                      DenseVertexSet<vid_t>& modified) {}

  virtual void init_v(const vertex_t v, value_t& value) = 0;

  virtual void iterate_begin(const FRAG_T& frag) {}

  virtual bool accumulate(value_t& a, value_t b) = 0;

  inline bool accumulate_to(vertex_t& v, value_t val) {
    return accumulate(deltas_[v], val);
  }

  virtual void priority(value_t& pri, const value_t& value,
                        const value_t& delta) = 0;

  virtual void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& oes) {}

  virtual void g_function(const FRAG_T& frag, const vertex_t v,
                          const value_t& value, const value_t& delta,
                          const adj_list_t& oes,
                          DenseVertexSet<vid_t>& modified) {}

  virtual value_t default_v() = 0;

  virtual value_t min_delta() = 0;

  // if u sends msg to v, then v depends on u
  inline void add_delta_dependency(vid_t u_gid, vertex_t v) {
    delta_dep_[v] = u_gid;
  }

  inline void mark_value_dependency(vertex_t u) { val_dep_[u] = delta_dep_[u]; }

  inline vid_t delta_parent_gid(vertex_t v) { return delta_dep_[v]; }

  inline vid_t value_parent_gid(vertex_t v) { return val_dep_[v]; }

  std::vector<value_t> DumpResult() {
    std::vector<value_t> result;

    for (auto v : values_.GetVertexRange()) {
      result.push_back(values_[v]);
    }
    return result;
  }

 protected:
  void Init(const CommSpec& comm_spec, const FRAG_T& frag, bool dependent,
            bool data_driven = false) {
    auto vertices = frag.Vertices();
    auto inner_vertices = frag.InnerVertices();

    values_.Init(inner_vertices);
    deltas_.Init(vertices, default_v());

    if (data_driven) {
      curr_modified_.Init(vertices);
      next_modified_.Init(vertices);
      curr_modified_.Clear();
      next_modified_.Clear();

      for (auto v : inner_vertices) {
        value_t value;
        init_v(v, value);
        values_[v] = value;

        value_t delta;
        init_c(frag, v, delta, curr_modified_);
        deltas_[v] = delta;
      }
    } else {
      for (auto v : inner_vertices) {
        value_t value;
        init_v(v, value);
        values_[v] = value;

        value_t delta;
        init_c(v, delta, frag);
        deltas_[v] = delta;
      }
    }

    if (dependent) {
      val_dep_.Init(inner_vertices);
      delta_dep_.Init(vertices);

      for (auto v : inner_vertices) {
        val_dep_[v] = frag.Vertex2Gid(v);
      }

      for (auto v : vertices) {
        delta_dep_[v] = frag.Vertex2Gid(v);
      }
    }

    uint64_t memory = 0, global_mem;
    if (!FLAGS_efile_update.empty()) {
      memory += sizeof(vid_t) * val_dep_.size();
      memory += sizeof(vid_t) * delta_dep_.size();
    }
    memory += curr_modified_.Range().size() / 64;
    memory += next_modified_.Range().size() / 64;
    memory += sizeof(value_t) * values_.size();
    memory += sizeof(value_t) * deltas_.size();

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

  VertexArray<vid_t, vid_t> val_dep_;
  VertexArray<vid_t, vid_t> delta_dep_;
  DenseVertexSet<vid_t> curr_modified_, next_modified_;
  VertexArray<value_t, vid_t> values_{};
  VertexArray<value_t, vid_t> deltas_{};
  bool batch_stage_{true};
  //  VertexArray<value_t, vid_t> priority_{};  //每个顶点对应的优先级
  template <typename APP_T>
  friend class AsyncWorker;
  template <typename APP_T>
  friend class IngressSyncWorker;
  template <typename APP_T>
  friend class IngressSyncSSSPWorker;
  template <typename APP_T>
  friend class IngressSyncPrWorker;
  template <typename APP_T>
  friend class IngressSyncTraversalWorker;
  template <typename APP_T>
  friend class IngressSyncIterWorker;
};

}  // namespace grape
#endif  // LIBGRAPE_LITE_GRAPE_APP_INGRESS_APP_BASE_H_
