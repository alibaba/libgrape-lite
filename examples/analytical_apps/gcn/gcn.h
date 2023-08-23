/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef EXAMPLES_ANALYTICAL_APPS_GCN_GCN_H_
#define EXAMPLES_ANALYTICAL_APPS_GCN_GCN_H_

#include <grape/grape.h>

#include "grape/app/inc_app_base.h"

namespace grape {
/**
 * @brief Context for the parallel version of GCN.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class GCNContext : public ContextBase<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  void Init(const FRAG_T& frag, ParallelMessageManager& messages,
            const CommSpec& comm_spec, int max_iter) {
    auto iv = frag.InnerVertices();
    auto vertices = frag.Vertices();

    iter = 0;

    srand(0);

    features.resize(max_iter + 1);

    for (int i = 0; i <= max_iter; i++) {
      features[i].Init(vertices);
    }

    // |V|,m0 x m0,m1 x m1,m2 x ... m2,1
    int final_features = 16;
    n_features = final_features;

    for (int i = 0; i < max_iter; i++) {
      n_features *= 2;
    }

    LOG(INFO) << "Generating features"
              << ", " << n_features << " features/vertex";
    LOG(INFO) << frag.GetTotalVerticesNum() * n_features * sizeof(float) /
                     1024 / 1024 / 1024.0
              << " GB.";
    parallel_for(size_t vid = 0; vid < iv.size(); vid++) {
      auto v = Vertex<vid_t>(vid);

      for (int i = 0; i < n_features; i++) {
        //        int sign = randi(0, 1) ? -1 : 1;
        //        auto val = sign * randi(1, 32);
        auto val = frag.GetId(v) + i;
        features[0][v].push_back(val);
      }
    }

    int m = n_features;

    for (int i = 0; i < max_iter && m > 1; i++) {
      // shape: m rows, next_m cols
      int next_m = m / 2;

      std::vector<std::vector<float>> weight_mtx;
      for (int col = 0; col < next_m; col++) {
        std::vector<float> weight_col(m);

        for (int row = 0; row < m; row++) {
          weight_col[row] = randi(1, 64);
        }

        // normalize
        float sum = 0;
        for (int row = 0; row < m; row++) {
          sum += weight_col[row];
        }

        for (int row = 0; row < m; row++) {
          weight_col[row] /= sum;
        }

        weight_mtx.push_back(weight_col);
      }

      weight_matrix_by_iter.push_back(weight_mtx);
      m = next_m;
    }
    this->max_iter = weight_matrix_by_iter.size();

    LOG(INFO) << "Max iterations: " << this->max_iter;

    uint64_t memory = 0, global_mem;

    if (FLAGS_efile_update.empty()) {  // BATCH
      auto rg = features[0].GetVertexRange();

      memory += sizeof(float) * n_features * rg.size();
    } else {  // INC
      auto rg = features[0].GetVertexRange();
      auto tmp = n_features;

      for (int i = 0; i < max_iter; ++i) {
        memory += sizeof(float) * tmp * rg.size();
        tmp /= 2;
      }
    }

    for (auto& e1 : weight_matrix_by_iter) {
      for (auto& e2 : e1) {
        memory += sizeof(float) * e2.size();
      }
    }

    // INC
    if (!FLAGS_efile_update.empty()) {
      memory += vertices.size() / 64;
      memory += vertices.size() / 64;
      memory += sizeof(float) * vals_before_update.size() * n_features;
    }

    Communicator communicator;
    communicator.InitCommunicator(comm_spec.comm());
    communicator.template Sum(memory, global_mem);

    if (comm_spec.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Mem: " << global_mem / 1024 / 1024 << " MB";
    }
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto iv = frag.InnerVertices();

    LOG(INFO) << this->features.size();
    for (auto v : iv) {
      auto& features = this->features[max_iter][v];
      os << frag.GetId(v);
      for (auto f : features) {
        os << " " << f;
      }
      os << std::endl;
    }
  }

  int randi(int lo, int hi) {
    int n = hi - lo + 1;
    int i = std::rand() % n;
    if (i < 0)
      i = -i;
    return lo + i;
  }

  std::vector<VertexArray<std::vector<float>, vid_t>> features;
  // column oriented
  std::vector<std::vector<std::vector<float>>> weight_matrix_by_iter;
  VertexArray<std::vector<float>, vid_t>
      vals_before_update;  // current values before update
  DenseVertexSet<vid_t> active_in, active_out;
  // the destinations are: edge in update file, edge in the updated graph
  VertexArray<std::vector<float>, vid_t> delta_vals;
  VertexArray<std::vector<float>, vid_t> reduced_features_old,
      reduced_features_new;
  int iter;
  int max_iter;
  int n_features;
};

template <typename FRAG_T>
class GCN : public IncAppBase<FRAG_T, GCNContext<FRAG_T>>,
            public ParallelEngine {
 public:
  // specialize the templated worker.
  INSTALL_INC_WORKER(GCN<FRAG_T>, GCNContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  /**
   * @brief Partial evaluation for GCN.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    messages.InitChannels(thread_num());
    messages.ForceContinue();
  }

  /**
   * @brief Incremental evaluation for GCN.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto& iter = ctx.iter;
    auto& weight_mtx = ctx.weight_matrix_by_iter[iter];
    auto& curr_features = ctx.features[iter];

    messages.template ParallelProcess<fragment_t, std::vector<float>>(
        thread_num(), frag,
        [&curr_features](int tid, vertex_t v, std::vector<float>& msg) {
          CHECK_EQ(curr_features[v].size(), msg.size());
          for (size_t i = 0; i < msg.size(); i++) {
            curr_features[v][i] += msg[i];
          }
        });

    if (iter >= ctx.max_iter) {
      return;
    } else {
      messages.ForceContinue();
    }
    LOG(INFO) << "Iter: " << iter;

    auto& next_features = ctx.features[iter + 1];
    auto feature_size_next = weight_mtx.size();

    ForEach(vertices, [&next_features, feature_size_next](int tid, vertex_t v) {
      next_features[v].clear();
      next_features[v].resize(feature_size_next, 0);
    });

    VertexArray<std::vector<float>, vid_t> reduced_features;

    reduced_features.Init(iv);

    // dim reduction
    ForEach(iv, [this, &reduced_features, &weight_mtx, &curr_features,
                 feature_size_next, iter](int tid, vertex_t v) {
      reduced_features[v].resize(feature_size_next);

      for (size_t col = 0; col < feature_size_next; col++) {
        auto& weight_col = weight_mtx[col];
        std::vector<float> tmp = curr_features[v];

        // To make incremental feasible, we have to evaluate relu before the
        // iteration instead of the end of iteration
        if (iter > 0) {
          for (float& i : tmp) {
            i = relu(i);
          }
        }

        auto sum = VectorSum(tmp, weight_col);

        reduced_features[v][col] = sum;
      }
    });

    // push features to neighbors
    // N.B. We don't implement self cycle to prevent implement complicated
    // incremental logic
    ForEach(iv,
            [&frag, &reduced_features, &next_features](int tid, vertex_t u) {
              auto oes = frag.GetOutgoingAdjList(u);

              for (auto e : oes) {
                auto v = e.neighbor;

                for (size_t i = 0; i < reduced_features[u].size(); i++) {
                  atomic_add(next_features[v][i], reduced_features[u][i]);
                }
              }
            });

    ForEach(ov, [&frag, &messages, &next_features](int tid, vertex_t v) {
      messages.Channels()[tid].SyncStateOnOuterVertex(frag, v,
                                                      next_features[v]);
    });

    iter++;
  }

  void AdjustPEval(const fragment_t& updated_frag,
                   const std::vector<std::pair<oid_t, oid_t>>& added_edges,
                   const std::vector<std::pair<oid_t, oid_t>>& deleted_edges,
                   context_t& ctx, message_manager_t& messages) {
    ctx.iter = 0;

    // FIXME: BUG SHOULD COPY FEATURES
    auto curr_n_features = ctx.n_features;
    auto vertices = updated_frag.Vertices();
    auto iv = updated_frag.InnerVertices();

    ctx.active_in.Init(vertices);
    ctx.active_out.Init(vertices);
    ctx.delta_vals.Init(vertices);

    ctx.reduced_features_old.Init(iv);
    ctx.reduced_features_new.Init(iv);

    for (auto& iter_vals : ctx.features) {
      VertexArray<std::vector<float>, vid_t> backup;
      backup.Init(vertices);

      for (auto v : updated_frag.InnerVertices()) {
        auto tmp = iter_vals[v];
        backup[v] = tmp;
      }

      for (auto v : updated_frag.OuterVertices()) {
        backup[v].resize(curr_n_features, 0);
      }

      iter_vals.Init(vertices);
      for (auto v : vertices) {
        iter_vals[v] = backup[v];
      }

      curr_n_features /= 2;
    }

    ctx.vals_before_update = ctx.features[0];

    messages.InitChannels(thread_num());
    messages.ForceContinue();
  }

  void AdjustIncEval(const fragment_t& updated_frag,
                     const std::vector<std::pair<oid_t, oid_t>>& added_edges,
                     const std::vector<std::pair<oid_t, oid_t>>& deleted_edges,
                     context_t& ctx, message_manager_t& messages) {
    auto vertices = updated_frag.Vertices();
    auto iv = updated_frag.InnerVertices();
    auto ov = updated_frag.OuterVertices();
    auto& iter = ctx.iter;
    auto& weight_mtx = ctx.weight_matrix_by_iter[iter];
    auto feature_size_next = weight_mtx.size();
    auto& delta_vals = ctx.delta_vals;
    auto& reduced_features_old = ctx.reduced_features_old;
    auto& reduced_features_new = ctx.reduced_features_new;

    messages.template ParallelProcess<fragment_t, std::vector<float>>(
        thread_num(), updated_frag,
        [&ctx, iter](int tid, vertex_t v, std::vector<float>& msg) {
          ctx.active_in.Insert(v);
          CHECK_EQ(ctx.features[iter][v].size(), msg.size());
          for (size_t i = 0; i < msg.size(); i++) {
            ctx.features[iter][v][i] += msg[i];
          }
        });

    if (iter >= ctx.max_iter) {
      return;
    } else {
      messages.ForceContinue();
    }
    LOG(INFO) << "Iter: " << iter;

    ForEach(vertices, [&delta_vals, feature_size_next](int tid, vertex_t v) {
      delta_vals[v].resize(feature_size_next, 0);
    });

    parallel_for(size_t i = 0; i < added_edges.size(); i++) {
      auto& e = added_edges[i];
      vertex_t u, v;
      CHECK(updated_frag.GetVertex(e.first, u));
      CHECK(updated_frag.GetVertex(e.second, v));
      ctx.active_in.Insert(u);
      ctx.active_out.Insert(v);
    }

    // FIXME:(Delete edges is not supported for multiple workers)
    if (updated_frag.fnum() == 1) {
      for (auto& e : deleted_edges) {
        vertex_t u, v;
        CHECK(updated_frag.GetVertex(e.first, u));
        CHECK(updated_frag.GetVertex(e.second, v));
        ctx.active_in.Insert(u);
        ctx.active_out.Insert(v);
        delta_vals[v].resize(feature_size_next, 0);
      }
    }

    ForEach(ctx.active_in, iv,
            [&reduced_features_old, &reduced_features_new, &updated_frag,
             &delta_vals, &ctx, feature_size_next](int tid, vertex_t u) {
              auto& r_features_old = reduced_features_old[u];
              auto& r_features_new = reduced_features_new[u];

              r_features_old.resize(feature_size_next, 0);
              r_features_new.resize(feature_size_next, 0);

              auto oes = updated_frag.GetOutgoingAdjList(u);

              for (auto e : oes) {
                auto v = e.neighbor;

                delta_vals[v].resize(feature_size_next, 0);
                ctx.active_out.Insert(v);
              }
            });

    // reduce curr_vals by multiplied weight matrix
    ForEach(ctx.active_in, iv,
            [this, &ctx, &reduced_features_old, &reduced_features_new,
             &weight_mtx, feature_size_next, iter](int tid, vertex_t v) {
              auto& r_features_old = reduced_features_old[v];
              auto& r_features_new = reduced_features_new[v];

              for (size_t col = 0; col < feature_size_next; col++) {
                auto& weight_col = weight_mtx[col];
                auto& tmp1 = ctx.vals_before_update[v];
                auto& tmp2 = ctx.features[iter][v];

                if (iter > 0) {
                  for (auto& i : tmp1) {
                    i = relu(i);
                  }

                  for (auto& i : tmp2) {
                    i = relu(i);
                  }
                }

                r_features_old[col] = VectorSum(tmp1, weight_col);
                r_features_new[col] = VectorSum(tmp2, weight_col);
              }
            });
    // Calculate the delta: extra messages and missing messages
    parallel_for(size_t i = 0; i < added_edges.size(); i++) {
      auto& e = added_edges[i];
      vertex_t u, v;
      CHECK(updated_frag.GetVertex(e.first, u));
      CHECK(updated_frag.GetVertex(e.second, v));

      CHECK(updated_frag.IsInnerVertex(u));
      for (size_t col = 0; col < feature_size_next; col++) {
        atomic_add(delta_vals[v][col], reduced_features_old[u][col]);
      }
    }
    // FIXME:(Delete edges is not supported for multiple workers)
    if (updated_frag.fnum() == 1) {
      parallel_for(size_t i = 0; i < deleted_edges.size(); i++) {
        auto& e = deleted_edges[i];
        vertex_t u, v;
        CHECK(updated_frag.GetVertex(e.first, u));
        CHECK(updated_frag.GetVertex(e.second, v));

        for (size_t col = 0; col < feature_size_next; col++) {
          CHECK_LT(col, delta_vals[v].size());
          atomic_add(delta_vals[v][col], -reduced_features_old[u][col]);
        }
      }
    }

    ForEach(ctx.active_in, iv,
            [&updated_frag, &delta_vals, &reduced_features_old,
             &reduced_features_new, feature_size_next](int tid, vertex_t u) {
              auto oes = updated_frag.GetOutgoingAdjList(u);

              for (auto e : oes) {
                auto v = e.neighbor;

                for (size_t col = 0; col < feature_size_next; col++) {
                  atomic_add(delta_vals[v][col],
                             reduced_features_new[u][col] -
                                 reduced_features_old[u][col]);
                }
              }
            });

    ForEach(ctx.active_out, ov,
            [&messages, &updated_frag, &delta_vals](int tid, vertex_t v) {
              messages.Channels()[tid].SyncStateOnOuterVertex(updated_frag, v,
                                                              delta_vals[v]);
            });

    ForEach(iv, [&ctx, iter](int tid, vertex_t v) {
      ctx.vals_before_update[v] = ctx.features[iter + 1][v];
    });

    ForEach(ctx.active_out, iv,
            [&ctx, &delta_vals, iter, feature_size_next](int tid, vertex_t v) {
              auto& next_features = ctx.features[iter + 1];
              CHECK_EQ(next_features[v].size(), delta_vals[v].size());
              CHECK_EQ(next_features[v].size(), feature_size_next);

              for (size_t col = 0; col < feature_size_next; col++) {
                next_features[v][col] += delta_vals[v][col];
              }
            });
    ctx.active_in.Clear();
    ctx.active_in.Swap(ctx.active_out);
    iter++;
  }

  template <class T>
  float VectorSum(const std::vector<T>& a, const std::vector<T>& b) {
    CHECK_EQ(a.size(), b.size());
    float sum = 0;

    for (size_t i = 0; i < a.size(); i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  float relu(float in) { return in > 0 ? in : 0; }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_GCN_GCN_H_
