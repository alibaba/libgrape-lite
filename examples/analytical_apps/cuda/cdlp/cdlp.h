/** Copyright 2023 Alibaba Group Holding Limited.

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

    labels.Init(vertices);
    label_acc.Init(iv);
    new_label.Init(iv, thrust::make_pair(0, false));

    hi_q.Init(iv.size());
    lo_q.Init(iv.size());

    d_lo_row_offset.resize(iv.size() + 1);
    d_hi_row_offset.resize(iv.size() + 1);

    if (frag.load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_row_offset.resize(iv.size() + 1);
    }

    for (auto v : iv) {
      labels[v] = frag.GetInnerVertexId(v);
      label_acc[v] = std::numeric_limits<label_t>::max();
    }

    for (auto v : ov) {
      labels[v] = frag.GetOuterVertexId(v);
    }
    labels.H2D();
    label_acc.H2D();

#ifdef PROFILING
    get_msg_time = 0;
    traversal_kernel_time = 0;
#endif

    messages.InitBuffer(  // N.B. pair padding
        (sizeof(thrust::pair<vid_t, label_t>)) * iv.size(),
        (sizeof(thrust::pair<vid_t, label_t>)) * 1);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    labels.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << labels[v] << std::endl;
    }
  }

  int step;
  int max_round;
  LoadBalancing lb;
  VertexArray<label_t, vid_t> labels;
  VertexArray<thrust::pair<label_t, bool>, vid_t> new_label;
  VertexArray<label_t, vid_t> label_acc;

  // first-round:
  thrust::device_vector<size_t> d_row_offset;
  thrust::device_vector<label_t> d_col_indices;
  thrust::device_vector<label_t> d_sorted_col_indices;
  Queue<vertex_t, vid_t> hi_q, lo_q;
  // low-degree: warp-level match_any_sync + reduce_sync
  thrust::device_vector<size_t> d_lo_row_offset;
  thrust::device_vector<label_t> d_lo_col_indices;
  // high-degree: block-level shm_ht + CMS + gm_ht
  thrust::device_vector<size_t> d_hi_row_offset;
  thrust::device_vector<label_t> d_hi_col_indices;
  thrust::device_vector<label_t> d_hi_label_hash;
  thrust::device_vector<uint32_t> d_hi_label_cnt;

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
  using size_type = size_t;

  static constexpr grape::MessageStrategy message_strategy =
      MessageStrategyTrait<FRAG_T::load_strategy>::message_strategy;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;
  // static constexpr bool need_build_device_vm = true;  // for debug

  void PropagateLabel_first_d(const fragment_t& frag, context_t& ctx,
                              message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_labels = ctx.labels.DeviceObject();
    auto* d_offsets = thrust::raw_pointer_cast(ctx.d_row_offset.data());

    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());

    thrust::device_vector<size_t> out_degree(iv.size());
    auto* d_out_degree = thrust::raw_pointer_cast(out_degree.data());

    ForEachWithIndex(stream, ws_iv,
                     [=] __device__(size_t idx, vertex_t v) mutable {
                       size_t degree = d_frag.GetLocalOutDegree(v);
                       degree += d_frag.GetLocalInDegree(v);
                       d_out_degree[idx] = degree;
                     });

    auto size = iv.size();

    InclusiveSum(d_out_degree, d_offsets + 1, size, stream.cuda_stream());
    stream.Sync();

    size_t esize = ctx.d_row_offset[size];
    ctx.d_col_indices.resize(esize, 0);
    ctx.d_sorted_col_indices.resize(esize, 0);
    auto* p_d_col_indices = thrust::raw_pointer_cast(ctx.d_col_indices.data());

    ForEachIncomingEdge(
        stream, d_frag, ws_iv,
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          size_t eid =
              d_offsets[u.GetValue()] + d_frag.GetIncomingEdgeIndex(u, nbr);
          p_d_col_indices[eid] = d_labels[v];
        },
        ctx.lb);
    ForEachOutgoingEdge(
        stream, d_frag, ws_iv,
        [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          size_t eid = d_offsets[u.GetValue()] + d_frag.GetLocalInDegree(u) +
                       d_frag.GetOutgoingEdgeIndex(u, nbr);
          p_d_col_indices[eid] = d_labels[v];
        },
        ctx.lb);
    stream.Sync();

    auto* local_labels = thrust::raw_pointer_cast(ctx.d_col_indices.data());

    {
      size_t num_segments = iv.size();
      size_t num_items = ctx.d_row_offset[num_segments];
      auto* p_d_col_indices =
          thrust::raw_pointer_cast(ctx.d_col_indices.data());
      auto* p_d_sorted_col_indices =
          thrust::raw_pointer_cast(ctx.d_sorted_col_indices.data());

      stream.Sync();
      local_labels =
          SegmentSort(p_d_col_indices, p_d_sorted_col_indices, d_offsets,
                      d_offsets + 1, num_items, num_segments);
    }

    auto d_new_label = ctx.new_label.DeviceObject();

    ForEachWithIndex(
        stream, ws_iv, [=] __device__(size_t idx, vertex_t v) mutable {
          idx = v.GetValue();
          size_t begin = d_offsets[idx], end = d_offsets[idx + 1];
          size_t size = end - begin;

          if (size > 0) {
            label_t new_label;
            label_t curr_label = local_labels[begin];
            int64_t curr_count = 1;
            label_t best_label = 0;
            int64_t best_count = 0;

            // Enumerate its neighbor to find MFL
            // TODO(mengke.mk) Single thread with severe load-imbalance.
            for (auto eid = begin + 1; eid < end; eid++) {
              if (local_labels[eid] != local_labels[eid - 1]) {
                if (curr_count > best_count ||
                    (curr_count == best_count && curr_label < best_label)) {
                  best_label = curr_label;
                  best_count = curr_count;
                }
                curr_label = local_labels[eid];
                curr_count = 1;
              } else {
                ++curr_count;
              }
            }

            if (curr_count > best_count ||
                (curr_count == best_count && curr_label < best_label)) {
              new_label = curr_label;
            } else {
              new_label = best_label;
            }

            if (new_label != d_labels[v]) {
              d_new_label[v].first = new_label;
              d_new_label[v].second = true;
              d_mm.template SendMsgThroughEdges(d_frag, v, new_label);
            } else {
              d_new_label[v].second = false;
            }
          }
        });
    stream.Sync();

    ctx.d_row_offset.resize(0);
    ctx.d_row_offset.shrink_to_fit();
    ctx.d_col_indices.resize(0);
    ctx.d_col_indices.shrink_to_fit();
    ctx.d_sorted_col_indices.resize(0);
    ctx.d_sorted_col_indices.shrink_to_fit();
  }

  void PropagateLabel_first_ud(const fragment_t& frag, context_t& ctx,
                               message_manager_t& messages) {
    double time = grape::GetCurrentTime();
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_labels = ctx.labels.DeviceObject();
    auto d_label_acc = ctx.label_acc.DeviceObject();
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;
    auto d_new_label = ctx.new_label.DeviceObject();

    WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());

    ForEachOutgoingEdge(
        stream, d_frag, ws_in,
        [=] __device__(const vertex_t& u, const nbr_t& nbr) mutable {
          vertex_t v = nbr.get_neighbor();
          label_t l = d_labels[v];
          label_t* slot = &d_label_acc[u];
          dev::atomicMin64(slot, l);
        },
        ctx.lb);

    ForEach(stream, ws_in, [=] __device__(vertex_t u) mutable {
      label_t new_label = d_label_acc[u];
      if (d_frag.GetLocalOutDegree(u) == 0) {
        return;
      }
      if (new_label != d_labels[u]) {
        d_new_label[u].first = new_label;
        d_new_label[u].second = true;
        if (isDirected) {
          d_mm.template SendMsgThroughEdges(d_frag, u, new_label);
        } else {
          d_mm.template SendMsgThroughOEdges(d_frag, u, new_label);
        }
      } else {
        d_new_label[u].second = false;
      }
    });
  }

  void PropagateLabel_hi(const fragment_t& frag, context_t& ctx,
                         message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_labels = ctx.labels.DeviceObject();
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;
    auto* d_offsets = thrust::raw_pointer_cast(ctx.d_hi_row_offset.data());
    auto* p_d_col_indices =
        thrust::raw_pointer_cast(ctx.d_hi_col_indices.data());
    auto& hi_q = ctx.hi_q;
    auto d_new_label = ctx.new_label.DeviceObject();

    WorkSourceArray<vertex_t> ws_iv(hi_q.data(), hi_q.size(stream));
    if (isDirected) {
      ForEachIncomingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetIncomingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid = d_offsets[u.GetValue()] + d_frag.GetLocalInDegree(u) +
                         d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    } else {
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    }

    auto* local_labels = thrust::raw_pointer_cast(ctx.d_hi_col_indices.data());
    auto* global_data = thrust::raw_pointer_cast(ctx.d_hi_label_hash.data());
    auto* label_cnt = thrust::raw_pointer_cast(ctx.d_hi_label_cnt.data());

    const label_t DFT_LABEL = std::numeric_limits<label_t>::max();
    thrust::fill(ctx.d_hi_label_hash.begin(), ctx.d_hi_label_hash.end(), -1);
    thrust::fill(ctx.d_hi_label_cnt.begin(), ctx.d_hi_label_cnt.end(), 0);

    int bucket_size = 2048;
    int cms_size = 2048;
    int cms_k = 1;
    int group_id = 0;
    int group_size = 1;
    int width = sizeof(label_t) / sizeof(uint32_t);

    ForEachWithIndexBlockShared(
        stream, ws_iv,
        [=] __device__(uint32_t * shm, size_t lane, size_t cid, size_t csize,
                       size_t cnum, size_t idx, vertex_t u) mutable {
          label_t* shmd = reinterpret_cast<label_t*>(shm);
          for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
            if (i < bucket_size) {
              shmd[i] = -1;
            } else if (i >= bucket_size * width) {
              shm[i] = 0;
            }
          }
          __syncthreads();

          idx = u.GetValue();
          size_t begin = d_offsets[idx], end = d_offsets[idx + 1];
          if (end == begin) {
            return;
          }

          dev::MFLCounter<label_t> counter;
          counter.init(shm, global_data, label_cnt, bucket_size, cms_size,
                       cms_k, group_id, group_size, width, -1);
          __shared__ label_t new_label;

          label_t max_label = 0;
          int ht_score = 0;
          int cms_score = 0;
          int gt_score = 0;

          // step 1: try CMS speculatively
          label_t local_label = DFT_LABEL;
          int sh_ht_freq = -1;
          int sh_cms_freq = -1;
          __syncthreads();
          for (auto eid = begin + threadIdx.x; eid < end; eid += blockDim.x) {
            auto l = local_labels[eid];
            int current_sh_ht_freq = counter.insert_shm_ht(l);
            int current_sh_cms_freq = -1;

            if (current_sh_ht_freq < 0) {
              current_sh_cms_freq = counter.insert_shm_cms(l);
            }

            // update locally
            int current = current_sh_ht_freq > current_sh_cms_freq
                              ? current_sh_ht_freq
                              : current_sh_cms_freq;
            int old = sh_ht_freq > sh_cms_freq ? sh_ht_freq : sh_cms_freq;
            if (current > old || current == old && local_label > l) {
              local_label = l;
              sh_ht_freq = current_sh_ht_freq;
              sh_cms_freq = current_sh_cms_freq;
            }
          }
          __syncthreads();

          // step 2: check whether our CMS works
          ht_score = sh_ht_freq > 0 ? sh_ht_freq : 0;
          cms_score = sh_cms_freq > 0 ? sh_cms_freq : 0;
          max_label = local_label;
          __syncthreads();
          max_label = dev::blockAllReduceMax(max_label);
          ht_score = dev::blockAllReduceMax(ht_score);
          cms_score = dev::blockAllReduceMax(cms_score);
          __syncthreads();
          if (ht_score > cms_score) {  // shared_memory is enough
            if (threadIdx.x == 0) {
              new_label = max_label;
            }
            __syncthreads();
            if (sh_ht_freq == ht_score) {
              dev::atomicMin64(&new_label, local_label);
            }
            __syncthreads();
          } else {
            // step 3: the bad case, we have to do it again
            label_t local_label = DFT_LABEL;
            int ht_freq = -1;
            for (auto eid = begin + threadIdx.x; eid < end; eid += blockDim.x) {
              auto l = local_labels[eid];
              int current_ht_freq = counter.query_shm_ht(l);

              if (current_ht_freq < 0) {
                assert(current_ht_freq != 0);
                current_ht_freq = counter.insert_global_ht(l, begin, end);
              }

              // update locally
              if (current_ht_freq > ht_freq ||
                  current_ht_freq == ht_freq && local_label > l) {
                local_label = l;
                ht_freq = current_ht_freq;
              }
            }
            __syncthreads();

            // step 4: now we have the true count.
            max_label = local_label;
            gt_score = ht_freq > 0 ? ht_freq : 0;
            max_label = dev::blockAllReduceMax(max_label);
            gt_score = dev::blockAllReduceMax(gt_score);
            __syncthreads();

            if (threadIdx.x == 0) {
              new_label = max_label;
            }
            __syncthreads();

            if (gt_score == ht_freq) {
              dev::atomicMin64(&new_label, local_label);
            }
            __syncthreads();
          }
          __syncthreads();

          // step 5: process the new label
          if (threadIdx.x == 0) {
            if (new_label != d_labels[u]) {
              d_new_label[u].first = new_label;
              d_new_label[u].second = true;
              if (isDirected) {
                d_mm.template SendMsgThroughEdges(d_frag, u, new_label);
              } else {
                d_mm.template SendMsgThroughOEdges(d_frag, u, new_label);
              }
            } else {
              d_new_label[u].second = false;
            }
          }
        });
  }

  void PropagateLabel_lo(const fragment_t& frag, context_t& ctx,
                         message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto* p_d_col_indices =
        thrust::raw_pointer_cast(ctx.d_lo_col_indices.data());
    auto d_labels = ctx.labels.DeviceObject();
    auto* d_offsets = thrust::raw_pointer_cast(ctx.d_lo_row_offset.data());
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;
    auto d_new_label = ctx.new_label.DeviceObject();
    auto& lo_q = ctx.lo_q;

    WorkSourceArray<vertex_t> ws_iv(lo_q.data(), lo_q.size(stream));
    if (isDirected) {
      ForEachIncomingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetIncomingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid = d_offsets[u.GetValue()] + d_frag.GetLocalInDegree(u) +
                         d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    } else {
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    }

    auto* local_labels = thrust::raw_pointer_cast(ctx.d_lo_col_indices.data());

    LaunchKernelFix(stream, ws_iv.size(), [=] __device__() mutable {
      __shared__ uint32_t V[256];
      __shared__ label_t labels[256];
      __shared__ uint32_t row_offset[256];
      __shared__ size_t start_pos[256];
      __shared__ uint32_t loaded_vertex[256];
      typedef cub::WarpScan<uint32_t, 32> WarpScan;
      __shared__ typename WarpScan::TempStorage temp_storage[8];
      auto tid = TID_1D;
      auto nthreads = TOTAL_THREADS_1D;
      auto warp_size = 32;
      auto warp_id = tid / warp_size;
      auto lane = tid % warp_size;
      auto n_warp = nthreads / warp_size;
      auto offset = (threadIdx.x / warp_size) * warp_size;

      auto total_size = ws_iv.size();

      auto local_start =
          (warp_id < (total_size % n_warp) ? warp_id : (total_size % n_warp)) +
          warp_id * (total_size / n_warp);
      auto local_size =
          (warp_id < (total_size % n_warp)) + (total_size / n_warp);
      for (size_t i = local_start; i < local_start + local_size;) {
        size_t v_size = 0;
        uint32_t chunk_size = 0;
        // engage more vertices in this warp.
        for (; i < local_start + local_size; i++) {
          auto u = ws_iv.GetWork(i);
          auto deg = d_offsets[u.GetValue() + 1] - d_offsets[u.GetValue()];
          assert(deg <= 32);
          if (deg == 0)
            continue;
          if (chunk_size + deg <= warp_size) {
            chunk_size += deg;
            if (lane == 0) {
              loaded_vertex[offset + v_size] = u.GetValue();
              row_offset[offset + v_size] = deg;
              start_pos[offset + v_size] = d_offsets[u.GetValue()];
            }
            v_size++;
          } else {
            break;
          }
          if (v_size >= 32) {
            break;
          }
        }
        __syncwarp();
        if (chunk_size == 0) {
          break;
        }
        uint32_t deg = 0;
        if (lane < v_size) {
          deg = row_offset[offset + lane];
        }
        __syncwarp();
        WarpScan(temp_storage[threadIdx.x / 32])
            .ExclusiveSum(deg, row_offset[threadIdx.x]);
        __syncwarp();
        assert(chunk_size <= warp_size);
        if (lane < chunk_size) {
          int loc = thrust::upper_bound(thrust::seq, row_offset + offset,
                                        row_offset + offset + v_size, lane) -
                    row_offset - offset - 1;
          V[offset + lane] = loaded_vertex[offset + loc];

          labels[offset + lane] = local_labels[start_pos[offset + loc] + lane -
                                               row_offset[offset + loc]];
        } else {
          V[offset + lane] = 0xffffffff;
          labels[offset + lane] = 0xffffffffffffffff;
        }
        __syncwarp();

        uint32_t active_mask = __ballot_sync(0xffffffff, lane < chunk_size);
        uint32_t vmask = __match_any_sync(active_mask, V[offset + lane]);
        uint32_t lmask = __match_any_sync(vmask, labels[offset + lane]);
        uint32_t count = __popc(lmask);
        uint32_t max_count = dev::reduce_max_sync(vmask, count);

        uint64_t candidate = 0xffffffffffffffff;
        if (lane < chunk_size) {
          assert(max_count != 0);
          candidate =
              max_count == count ? labels[offset + lane] : 0xffffffffffffffff;
        }
        __syncwarp();
        uint64_t min_candidate = dev::reduce_min_sync(vmask, candidate);
        label_t new_label = min_candidate;
        uint32_t leader = __ffs(vmask) - 1;

        if (lane < chunk_size) {
          assert(V[offset + lane] != 0xffffffff);
          vertex_t v(V[offset + lane]);
          assert(d_frag.IsInnerVertex(v));
          if (lane == leader && vmask != 0) {
            // the 0-idx edge will process the answer
            if (new_label != d_labels[v]) {
              d_new_label[v].first = new_label;
              d_new_label[v].second = true;
              if (isDirected) {
                d_mm.template SendMsgThroughEdges(d_frag, v, new_label);
              } else {
                d_mm.template SendMsgThroughOEdges(d_frag, v, new_label);
              }
            } else {
              d_new_label[v].second = false;
            }
          }
        }
        __syncwarp();
      }
    });
  }

  void Update(const fragment_t& frag, context_t& ctx,
              message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_labels = ctx.labels.DeviceObject();
    auto d_new_label = ctx.new_label.DeviceObject();

    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());

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
    auto d_lo_q = ctx.lo_q.DeviceObject();
    auto d_hi_q = ctx.hi_q.DeviceObject();
    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());

    thrust::device_vector<size_t> lo_out_degree(iv.size());
    thrust::device_vector<size_t> hi_out_degree(iv.size());
    auto* d_lo_out_degree = thrust::raw_pointer_cast(lo_out_degree.data());
    auto* d_hi_out_degree = thrust::raw_pointer_cast(hi_out_degree.data());

    ++ctx.step;
    if (ctx.step > ctx.max_round) {
      return;
    } else {
      messages.ForceContinue();
    }

    bool isDirected = (frag.load_strategy == grape::LoadStrategy::kBothOutIn);
    if (isDirected) {
      PropagateLabel_first_d(frag, ctx, messages);
    } else {
      PropagateLabel_first_ud(frag, ctx, messages);
    }
    Update(frag, ctx, messages);
    stream.Sync();

    ForEachWithIndex(stream, ws_iv,
                     [=] __device__(size_t idx, vertex_t v) mutable {
                       size_t degree = 0;
                       degree = d_frag.GetLocalOutDegree(v);
                       if (isDirected) {
                         degree += d_frag.GetLocalInDegree(v);
                       }
                       if (degree > 32) {
                         d_hi_q.Append(v);
                         d_hi_out_degree[idx] = degree;
                         d_lo_out_degree[idx] = 0;
                       } else {
                         d_lo_q.Append(v);
                         d_lo_out_degree[idx] = degree;
                         d_hi_out_degree[idx] = 0;
                       }
                     });

    auto* pd_lo_row_offset =
        thrust::raw_pointer_cast(ctx.d_lo_row_offset.data());
    auto* pd_hi_row_offset =
        thrust::raw_pointer_cast(ctx.d_hi_row_offset.data());
    auto size = iv.size();

    InclusiveSum(d_lo_out_degree, pd_lo_row_offset + 1, size,
                 stream.cuda_stream());
    InclusiveSum(d_hi_out_degree, pd_hi_row_offset + 1, size,
                 stream.cuda_stream());
    stream.Sync();

    size_t lo_size = ctx.d_lo_row_offset[size];
    size_t hi_size = ctx.d_hi_row_offset[size];

    ctx.d_lo_col_indices.resize(lo_size, 0);

    ctx.d_hi_label_hash.resize(hi_size, 0);
    ctx.d_hi_col_indices.resize(hi_size, 0);
    ctx.d_hi_label_cnt.resize(hi_size, 0);
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

    PropagateLabel_lo(frag, ctx, messages);
    PropagateLabel_hi(frag, ctx, messages);
    Update(frag, ctx, messages);
  }
};
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_
