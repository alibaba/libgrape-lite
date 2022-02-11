/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_

#ifdef __CUDACC__
#include <algorithm>

#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class CDLPContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using label_t = oid_t;

  explicit CDLPContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~CDLPContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "CDLP kernel time: " << traversal_kernel_time * 1000;
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config, int max_round) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    this->step = 0;
    this->max_round = max_round;
    this->lb = app_config.lb;

    for (auto v : iv) {
      oenum += this->fragment().GetOutgoingAdjList(v).Size();
    }

    labels.Init(vertices);
    new_label.Init(iv, thrust::make_pair(0, false));

    h_row_offset.resize(iv.size() + 1);
    d_row_offset.resize(iv.size() + 1);

    h_col_indices.resize(oenum);
    d_col_indices.resize(oenum);

    for (auto v : iv) {
      labels[v] = frag.GetInnerVertexId(v);
    }

    for (auto v : ov) {
      labels[v] = frag.GetOuterVertexId(v);
    }
    labels.H2D();

#ifdef PROFILING
    get_msg_time = 0;
    traversal_kernel_time = 0;
#endif

    messages.InitBuffer(100 * 1024 * 1024, 100 * 1024 * 1024);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    labels.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << labels[v] << std::endl;
    }
  }

  size_t oenum = 0;
  int step;
  int max_round;
  LoadBalancing lb;
  VertexArray<label_t, vid_t> labels;
  VertexArray<thrust::pair<label_t, bool>, vid_t> new_label;
  pinned_vector<int> h_row_offset;
  pinned_vector<label_t> h_col_indices;
  thrust::device_vector<int> d_row_offset;
  thrust::device_vector<label_t> d_col_indices;

#ifdef PROFILING
  double get_msg_time;
  double traversal_kernel_time;
#endif
};

template <typename FRAG_T>
class CDLP : public GPUAppBase<FRAG_T, CDLPContext<FRAG_T>>,
             public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(CDLP<FRAG_T>, CDLPContext<FRAG_T>, FRAG_T)
  using label_t = typename context_t::label_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;

  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEachHost(const VertexRange<VID_T>& range,
                          const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    std::atomic<VID_T> cur(range.begin_value());
    VID_T end = range.end_value();

    for (uint32_t i = 0; i < threads.size(); ++i) {
      threads[i] = std::thread(
          [&cur, chunk_size, &iter_func, end](uint32_t tid) {
            while (true) {
              VID_T cur_beg = std::min(cur.fetch_add(chunk_size), end);
              VID_T cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              VertexRange<VID_T> cur_range(cur_beg, cur_end);
              for (auto u : cur_range) {
                iter_func(tid, u);
              }
            }
          },
          i);
    }

    for (auto& thrd : threads) {
      thrd.join();
    }
  }

  void PropagateLabel(const fragment_t& frag, context_t& ctx,
                      message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());
    auto* p_d_col_indices = thrust::raw_pointer_cast(ctx.d_col_indices.data());
    auto d_labels = ctx.labels.DeviceObject();

    ForEachOutgoingEdge(
        stream, d_frag, ws_iv,
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          size_t eid = d_frag.GetOutgoingEdgeIndex(nbr);

          p_d_col_indices[eid] = d_labels[v];
        },
        ctx.lb);

    CHECK_CUDA(
        cudaMemcpyAsync(thrust::raw_pointer_cast(ctx.h_col_indices.data()),
                        thrust::raw_pointer_cast(ctx.d_col_indices.data()),
                        sizeof(label_t) * ctx.h_col_indices.size(),
                        cudaMemcpyDeviceToHost, stream.cuda_stream()));

    stream.Sync();

    {
      // TODO(mengke): A hybrid segmented sort. We may sort high-degree vertices
      // on GPU, sort relative low-degree vertices on CPU
#ifdef PROFILING
      auto begin = grape::GetCurrentTime();
#endif
      ForEachHost(
          iv,
          [&ctx](int tid, vertex_t v) {
            auto idx = v.GetValue();
            size_t begin = ctx.h_row_offset[idx],
                   end = ctx.h_row_offset[idx + 1];

            std::sort(ctx.h_col_indices.begin() + begin,
                      ctx.h_col_indices.begin() + end);
          },
          2048);
#ifdef PROFILING
      VLOG(1) << "Sort time: " << grape::GetCurrentTime() - begin;
#endif
    }

    CHECK_CUDA(
        cudaMemcpyAsync(thrust::raw_pointer_cast(ctx.d_col_indices.data()),
                        thrust::raw_pointer_cast(ctx.h_col_indices.data()),
                        sizeof(label_t) * ctx.h_col_indices.size(),
                        cudaMemcpyHostToDevice, stream.cuda_stream()));

    auto* d_offsets = thrust::raw_pointer_cast(ctx.d_row_offset.data());
    auto* local_labels = thrust::raw_pointer_cast(ctx.d_col_indices.data());

    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
    int n_vertices = iv.size();

    auto d_new_label = ctx.new_label.DeviceObject();

    ForEachWithIndex(
        stream, ws_in, [=] __device__(size_t idx, vertex_t v) mutable {
          size_t begin = d_offsets[idx], end = d_offsets[idx + 1];
          size_t size = end - begin;

          if (size > 0) {
            label_t new_label;
            label_t curr_label = local_labels[begin];
            int curr_count = 1;
            label_t best_label = 0;
            int best_count = 0;

            for (auto eid = begin + 1; eid < end; eid++) {
              if (local_labels[eid] != local_labels[eid - 1]) {
                if (curr_count > best_count) {
                  best_label = curr_label;
                  best_count = curr_count;
                }
                curr_label = local_labels[eid];
                curr_count = 1;
              } else {
                ++curr_count;
              }
            }

            if (curr_count > best_count) {
              new_label = curr_label;
            } else {
              new_label = best_label;
            }

            if (new_label != d_labels[v]) {
              d_new_label[v].first = new_label;
              d_new_label[v].second = true;
              d_mm.template SendMsgThroughOEdges(d_frag, v, new_label);
            } else {
              d_new_label[v].second = false;
            }
          }
        });

    ForEach(stream, ws_iv, [=] __device__(vertex_t v) mutable {
      if (d_new_label[v].second) {
        d_labels[v] = d_new_label[v].first;
      }
    });
  }

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto d_labels = ctx.labels.DeviceObject();
    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());
    WorkSourceRange<vertex_t> ws_ov(*ov.begin(), ov.size());
    thrust::device_vector<int> out_degree(iv.size());
    auto* d_out_degree = thrust::raw_pointer_cast(out_degree.data());

    ++ctx.step;
    if (ctx.step > ctx.max_round) {
      return;
    } else {
      messages.ForceContinue();
    }

    ForEachWithIndex(stream, ws_iv,
                     [=] __device__(size_t idx, vertex_t v) mutable {
                       d_out_degree[idx] = d_frag.GetLocalOutDegree(v);
                     });

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto* p_d_row_offset = thrust::raw_pointer_cast(ctx.d_row_offset.data());
    auto size = iv.size();

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_out_degree, p_d_row_offset + 1,
                                             size, stream.cuda_stream()));
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_out_degree, p_d_row_offset + 1,
                                             size, stream.cuda_stream()));
    CHECK_CUDA(cudaFree(d_temp_storage));

    CHECK_CUDA(
        cudaMemcpyAsync(thrust::raw_pointer_cast(ctx.h_row_offset.data()),
                        thrust::raw_pointer_cast(ctx.d_row_offset.data()),
                        sizeof(int) * ctx.h_row_offset.size(),
                        cudaMemcpyDeviceToHost, stream.cuda_stream()));

    PropagateLabel(frag, ctx, messages);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_labels = ctx.labels.DeviceObject();

    ctx.step++;

    // receive messages and set labels
    {
      messages.template ParallelProcess<dev_fragment_t, label_t>(
          d_frag, [=] __device__(vertex_t u, label_t msg) mutable {
            d_labels[u] = msg;
          });
    }

    if (ctx.step > ctx.max_round) {
      return;
    } else {
      messages.ForceContinue();
    }

    PropagateLabel(frag, ctx, messages);
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_
