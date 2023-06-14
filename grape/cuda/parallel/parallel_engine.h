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

#ifndef GRAPE_CUDA_PARALLEL_PARALLEL_ENGINE_H_
#define GRAPE_CUDA_PARALLEL_PARALLEL_ENGINE_H_

#include <cuda_profiler_api.h>

#include <unordered_set>
#pragma push
#pragma diag_suppress = initialization_not_reachable
#include <thrust/binary_search.h>

#include <cub/cub.cuh>
#include <moderngpu/kernel_sortedsearch.hxx>
#pragma pop

#include "grape/config.h"
#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/dev_utils.h"
#include "grape/cuda/utils/launcher.h"
#include "grape/cuda/utils/shared_value.h"
#include "grape/cuda/utils/sorted_search.h"
#include "grape/cuda/utils/work_source.h"

// TODO(liang): we may split this to multiple headers
namespace grape {
namespace cuda {

enum class EdgeDirection {
  kOutgoing,
  kIncoming,
  kOutgoingInner,
  kOutgoingOuter,
  kIncomingInner,
  kIncomingOuter
};

enum class LoadBalancing { kCMOld, kCM, kWarp, kCTA, kStrict, kNone };

LoadBalancing ParseLoadBalancing(const std::string& s_lb) {
  if (s_lb == "cmold" || s_lb == "CMOLD") {
    return LoadBalancing::kCMOld;
  } else if (s_lb == "cm" || s_lb == "CM") {
    return LoadBalancing::kCM;
  } else if (s_lb == "cta" || s_lb == "CTA") {
    return LoadBalancing::kCTA;
  } else if (s_lb == "wm" || s_lb == "WM") {
    return LoadBalancing::kWarp;
  } else if (s_lb == "strict" || s_lb == "STRICT") {
    return LoadBalancing::kStrict;
  } else if (s_lb == "none" || s_lb == "NONE") {
    return LoadBalancing::kNone;
  } else {
    LOG(FATAL) << "Invalid lb: " + s_lb;
    return LoadBalancing::kNone;
  }
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEach(const Stream& stream, const WORK_SOURCE_T& work_source,
                    FUNC_T func, Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  LaunchKernel(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (size_t i = 0 + tid; i < work_source.size(); i += nthreads) {
          auto work = work_source.GetWork(i);

          f(work, args...);
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndexWarpShared(const Stream& stream,
                                       const WORK_SOURCE_T& work_source,
                                       FUNC_T func, Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  LaunchKernelFix(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        __shared__ uint32_t shm[8192];
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;
        auto warp_size = 32;
        auto warp_id = tid / warp_size;
        auto lane = tid % warp_size;
        auto n_warp = nthreads / warp_size;

        for (size_t i = 0 + warp_id; i < work_source.size(); i += n_warp) {
          auto work = work_source.GetWork(i);

          f(shm, lane, warp_id, warp_size, n_warp, i, work,
            args...);  // A warp will do this
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndexBlockShared(const Stream& stream,
                                        const WORK_SOURCE_T& work_source,
                                        FUNC_T func, Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  LaunchKernelFix(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        __shared__ uint32_t shm[8192];
        auto tid = TID_1D;
        // auto nthreads = TOTAL_THREADS_1D;
        auto block_size = blockDim.x;
        auto block_id = blockIdx.x;
        auto lane = tid;
        auto n_block = gridDim.x;

        for (size_t i = 0 + block_id; i < work_source.size(); i += n_block) {
          auto work = work_source.GetWork(i);

          __syncthreads();
          f(shm, lane, block_id, block_size, n_block, i, work,
            args...);  // A warp will do this
          __syncthreads();
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndexWarp(const Stream& stream,
                                 const WORK_SOURCE_T& work_source, FUNC_T func,
                                 Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  LaunchKernelFix(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;
        auto warp_size = 32;
        auto warp_id = tid / warp_size;
        auto lane = tid % warp_size;
        auto n_warp = nthreads / warp_size;

        for (size_t i = 0 + warp_id; i < work_source.size(); i += n_warp) {
          auto work = work_source.GetWork(i);

          f(lane, i, work, args...);  // A warp will do this
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndexWarpDynamic(const Stream& stream,
                                        const WORK_SOURCE_T& work_source,
                                        FUNC_T func, Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  thrust::device_vector<unsigned long long int> ticket;  // NOLINT
  ticket.resize(1, 256 * 256 / 32);
  auto* d_ticket = thrust::raw_pointer_cast(ticket.data());
  LaunchKernelFix(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        auto tid = TID_1D;
        // auto nthreads = TOTAL_THREADS_1D;
        auto warp_size = 32;
        auto warp_id = tid / warp_size;
        auto lane = tid % warp_size;

        for (size_t i = 0 + warp_id; i < work_source.size();) {
          auto work = work_source.GetWork(i);

          f(lane, i, work, args...);  // A warp will do this
          __syncwarp();
          if (lane == 0) {
            i = atomicAdd(d_ticket, 1ll);
          }
          __syncwarp();
          i = __shfl_sync(0xffffffff, i, 0);
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndexBlock(const Stream& stream,
                                  const WORK_SOURCE_T& work_source, FUNC_T func,
                                  Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  LaunchKernelFix(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        // auto tid = TID_1D;
        // auto nthreads = TOTAL_THREADS_1D;
        // auto block_size = blockDim.x;
        auto block_id = blockIdx.x;
        auto lane = threadIdx.x;
        auto n_block = gridDim.x;

        for (size_t i = 0 + block_id; i < work_source.size(); i += n_block) {
          auto work = work_source.GetWork(i);

          __syncthreads();
          f(lane, i, work, args...);  // A blk will do this
          __syncthreads();
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndexBlockDynamic(const Stream& stream,
                                         const WORK_SOURCE_T& work_source,
                                         FUNC_T func, Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  thrust::device_vector<unsigned long long int> ticket;  // NOLINT
  ticket.resize(1, 256);
  auto* d_ticket = thrust::raw_pointer_cast(ticket.data());
  LaunchKernelFix(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        __shared__ size_t shared_idx;
        auto block_id = blockIdx.x;
        auto lane = threadIdx.x;
        // auto n_block = gridDim.x;

        for (size_t i = 0 + block_id; i < work_source.size();) {
          auto work = work_source.GetWork(i);

          __syncthreads();
          f(lane, i, work, args...);  // A warp will do this
          __syncthreads();
          if (lane == 0) {
            shared_idx = atomicAdd(d_ticket, 1ll);
          }
          __syncthreads();
          i = shared_idx;
        }
      },
      func, args...);
}

template <typename WORK_SOURCE_T, typename FUNC_T, typename... Args>
inline void ForEachWithIndex(const Stream& stream,
                             const WORK_SOURCE_T& work_source, FUNC_T func,
                             Args... args) {
  if (work_source.size() == 0) {
    return;
  }
  LaunchKernel(
      stream, work_source.size(),
      [=] __device__(FUNC_T f, Args... args) mutable {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (size_t i = 0 + tid; i < work_source.size(); i += nthreads) {
          auto work = work_source.GetWork(i);

          f(i, work, args...);
        }
      },
      func, args...);
}

template <typename VID_T, typename METADATA_T>
struct VertexMetadata {
  Vertex<VID_T> vertex;
  METADATA_T metadata;

  DEV_HOST VertexMetadata() {}

  DEV_HOST ~VertexMetadata() {}

  DEV_INLINE void set_vertex(const Vertex<VID_T>& v) { vertex = v; }

  DEV_INLINE void set_metadata(const METADATA_T& data) { metadata = data; }
};

template <typename VID_T>
struct VertexMetadata<VID_T, grape::EmptyType> {
  union {
    Vertex<VID_T> vertex;
    grape::EmptyType metadata;
  };

  DEV_HOST VertexMetadata<VID_T, grape::EmptyType>() {}

  DEV_HOST ~VertexMetadata<VID_T, grape::EmptyType>() {}

  DEV_INLINE void set_vertex(const Vertex<VID_T>& v) { vertex = v; }

  DEV_INLINE void set_metadata(const grape::EmptyType&) {}

  __device__ VertexMetadata<VID_T, grape::EmptyType>(
      const VertexMetadata<VID_T, grape::EmptyType>& rhs) {
    vertex = rhs.vertex;
  }

  __device__ VertexMetadata<VID_T, grape::EmptyType>& operator=(
      const VertexMetadata<VID_T, grape::EmptyType>& rhs) {
    if (this == &rhs) {
      return *this;
    }
    vertex = rhs.vertex;
    return *this;
  }
};

template <typename FRAG_T, EdgeDirection ed>
DEV_HOST_INLINE typename FRAG_T::adj_list_t GetAdjList(
    FRAG_T& frag, typename FRAG_T::vertex_t v) {
  switch (ed) {
  case EdgeDirection::kOutgoing: {
    return frag.GetOutgoingAdjList(v);
  }
  case EdgeDirection::kOutgoingInner: {
    return frag.GetOutgoingInnerVertexAdjList(v);
  }
  case EdgeDirection::kOutgoingOuter: {
    return frag.GetOutgoingOuterVertexAdjList(v);
  }
  case EdgeDirection::kIncoming: {
    return frag.GetIncomingAdjList(v);
  }
  case EdgeDirection::kIncomingInner: {
    return frag.GetIncomingInnerVertexAdjList(v);
  }
  case EdgeDirection::kIncomingOuter: {
    return frag.GetIncomingOuterVertexAdjList(v);
  }
  default:
    assert(false);
  }
  return {};
}

template <typename FRAG_T, EdgeDirection ed>
DEV_HOST_INLINE typename FRAG_T::const_adj_list_t GetAdjList(
    const FRAG_T& frag, typename FRAG_T::vertex_t v) {
  switch (ed) {
  case EdgeDirection::kOutgoing: {
    return frag.GetOutgoingAdjList(v);
  }
  case EdgeDirection::kOutgoingInner: {
    return frag.GetOutgoingInnerVertexAdjList(v);
  }
  case EdgeDirection::kOutgoingOuter: {
    return frag.GetOutgoingOuterVertexAdjList(v);
  }
  case EdgeDirection::kIncoming: {
    return frag.GetIncomingAdjList(v);
  }
  case EdgeDirection::kIncomingInner: {
    return frag.GetIncomingInnerVertexAdjList(v);
  }
  case EdgeDirection::kIncomingOuter: {
    return frag.GetIncomingOuterVertexAdjList(v);
  }
  default:
    assert(false);
  }
  return {};
}

template <typename VID_T, typename EDATA_T, const int WARPS_PER_TB,
          typename TMetaData>
struct warp_np {
  uint32_t owner[WARPS_PER_TB];
  const Nbr<VID_T, EDATA_T>* start[WARPS_PER_TB];
  VID_T size[WARPS_PER_TB];
  TMetaData meta_data[WARPS_PER_TB];
};

template <typename VID_T, typename EDATA_T, typename TMetaData>
struct tb_np {
  uint32_t owner;
  const Nbr<VID_T, EDATA_T>* start;
  VID_T size;
  TMetaData meta_data;
};

struct empty_np {};

template <typename ts_type, typename TTB, typename TWP, typename TFG = empty_np>
union np_shared {
  DEV_HOST np_shared() {}
  DEV_HOST ~np_shared() {}
  // for scans
  ts_type temp_storage;

  // for tb-level np
  TTB tb;

  // for warp-level np
  TWP warp;

  TFG fg;
};

/*
 * @brief A structure representing a scheduled chunk of work
 */
template <typename VID_T, typename EDATA_T, typename TMetaData>
struct np_local {
  __device__ np_local() {}
  __device__ ~np_local() {}

  __device__ np_local(const ConstAdjList<VID_T, EDATA_T>& adjlist,
                      const TMetaData& metadata)
      : start(adjlist.begin_pointer()),
        size(adjlist.Size()),
        meta_data(metadata) {}

  const Nbr<VID_T, EDATA_T>* start{};
  VID_T size{};
  TMetaData meta_data;
};

// TODO(liang): revisit CTAWorkScheduler and #define NO_CTA_WARP_INTRINSICS
template <typename VID_T, typename EDATA_T, typename TMetaData>
struct CTAWorkScheduler {
  template <typename TWork>
  __device__ __forceinline__ static void schedule(
      np_local<VID_T, EDATA_T, TMetaData>& np_local, TWork work) {
    const int WP_SIZE = CUB_PTX_WARP_THREADS;
    const int TB_SIZE = blockDim.x;

    const int NP_WP_CROSSOVER = WP_SIZE;
    const int NP_TB_CROSSOVER = TB_SIZE;

#ifndef NO_CTA_WARP_INTRINSICS
    typedef union np_shared<empty_np, tb_np<VID_T, EDATA_T, TMetaData>,
                            empty_np>
        np_shared_type;
#else
    typedef union np_shared<empty_np, tb_np<VID_T, EDATA_T, TMetaData>,
                            warp_np<VID_T, EDATA_T, 32, TMetaData>>
        np_shared_type;   // 32 is max number of warps in block
#endif

    __shared__ np_shared_type np_shared;

    if (threadIdx.x == 0) {
      np_shared.tb.owner = TB_SIZE + 1;
    }

    __syncthreads();

    //
    // First scheduler: processing high-degree work items using the entire block
    //
    while (true) {
      if (np_local.size >= NP_TB_CROSSOVER) {
        // 'Elect' one owner for the entire thread block
        np_shared.tb.owner = threadIdx.x;
      }

      __syncthreads();

      if (np_shared.tb.owner == TB_SIZE + 1) {
        // No owner was elected, i.e. no high-degree work items remain

#ifndef NO_CTA_WARP_INTRINSICS
        // No need to sync threads before moving on to WP scheduler
        // because it does not use shared memory
#else
        __syncthreads();  // Necessary do to the shared memory union used by
                          // both TB and WP schedulers
#endif
        break;
      }

      if (np_shared.tb.owner == threadIdx.x) {
        // This thread is the owner
        np_shared.tb.start = np_local.start;
        np_shared.tb.size = np_local.size;
        np_shared.tb.meta_data = np_local.meta_data;

        // Mark this work-item as processed for future schedulers
        np_local.start = nullptr;
        np_local.size = 0;
      }

      __syncthreads();

      auto start = np_shared.tb.start;
      auto size = np_shared.tb.size;
      auto meta_data = np_shared.tb.meta_data;

      if (np_shared.tb.owner == threadIdx.x) {
        np_shared.tb.owner = TB_SIZE + 1;
      }

      // Use all threads in thread block to execute individual work
      for (int ii = threadIdx.x; ii < size; ii += TB_SIZE) {
        work(*(start + ii), meta_data);
      }

      __syncthreads();
    }

    //
    // Second scheduler: tackle medium-degree work items using the warp
    //
#ifdef NO_CTA_WARP_INTRINSICS
    const int warp_id = threadIdx.x / WP_SIZE;
#endif
    const int lane_id = cub::LaneId();

    while (__any_sync(0xffffffff, np_local.size >= NP_WP_CROSSOVER)) {
#ifndef NO_CTA_WARP_INTRINSICS
      // Compete for work scheduling
      int mask =
          __ballot_sync(0xffffffff, np_local.size >= NP_WP_CROSSOVER ? 1 : 0);
      // Select a deterministic winner
      int leader = __ffs(mask) - 1;

      // Broadcast data from the leader
      auto start =
          cub::ShuffleIndex<WP_SIZE>(np_local.start, leader, 0xffffffff);
      auto size = cub::ShuffleIndex<WP_SIZE>(np_local.size, leader, 0xffffffff);
      auto meta_data =
          cub::ShuffleIndex<WP_SIZE>(np_local.meta_data, leader, 0xffffffff);

      if (leader == lane_id) {
        // Mark this work-item as processed
        np_local.start = nullptr;
        np_local.size = 0;
      }
#else
      if (np_local.size >= NP_WP_CROSSOVER) {
        // Again, race to select an owner for warp
        np_shared.warp.owner[warp_id] = lane_id;
      }

      cub::WARP_SYNC(0xffffffff);

      if (np_shared.warp.owner[warp_id] == lane_id) {
        // This thread is owner
        np_shared.warp.start[warp_id] = np_local.start;
        np_shared.warp.size[warp_id] = np_local.size;
        np_shared.warp.meta_data[warp_id] = np_local.meta_data;

        // Mark this work-item as processed
        np_local.start = nullptr;
        np_local.size = 0;
      }

      cub::WARP_SYNC(0xffffffff);

      auto start = np_shared.warp.start[warp_id];
      auto size = np_shared.warp.size[warp_id];
      auto meta_data = np_shared.warp.meta_data[warp_id];
#endif

      for (int ii = lane_id; ii < size; ii += WP_SIZE) {
        work(*(start + ii), meta_data);
      }

      // cub::WARP_SYNC(0xffffffff);
    }

    __syncthreads();

    //
    // Third scheduler: tackle all work-items with size < 32 serially
    //
    // We did not implement the FG (Finegrained) scheduling for simplicity
    // It is possible to disable this scheduler by setting NP_WP_CROSSOVER to 0
    while (__any_sync(0xffffffff, np_local.size > 0)) {
      int mask = __ballot_sync(0xffffffff, np_local.size > 0 ? 1 : 0);
      int leader = __ffs(mask) - 1;

      auto start = cub::ShuffleIndex<WP_SIZE>(np_local.start, leader, mask);
      auto size = cub::ShuffleIndex<WP_SIZE>(np_local.size, leader, mask);
      auto meta_data =
          cub::ShuffleIndex<WP_SIZE>(np_local.meta_data, leader, mask);

      if (leader == lane_id) {
        np_local.start = nullptr;
        np_local.size = 0;
      }

      if (lane_id < size) {
        work(*(start + lane_id), meta_data);
      }
    }
  }
};

template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
          typename EDGE_OP, EdgeDirection ed>
DEV_INLINE void LBNONE(const FRAG_T& dev_frag, const WORK_SOURCE_T& work_source,
                       ASSIGN_OP assign_op, EDGE_OP op) {
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;

  auto tid = TID_1D;
  auto nthreads = TOTAL_THREADS_1D;
  auto size = work_source.size();
  using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;
  using vertex_metadata_t = VertexMetadata<vid_t, metadata_t>;

  for (size_t i = 0 + tid; i < size; i += nthreads) {
    auto v = work_source.GetWork(i);
    auto adj_list = GetAdjList<FRAG_T, ed>(dev_frag, v);
    vertex_metadata_t vm;

    vm.set_vertex(v);
    vm.set_metadata(assign_op(v));

    for (size_t j = 0; j < adj_list.Size(); j++) {
      op(vm, adj_list.begin_pointer()[j]);
    }
  }
}

template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
          typename EDGE_OP, EdgeDirection ed>
DEV_INLINE void LBCMOld(const FRAG_T& dev_frag,
                        const WORK_SOURCE_T& work_source, ASSIGN_OP assign_op,
                        EDGE_OP op) {
  using nbr_t = typename FRAG_T::nbr_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;

  extern __shared__ char shared_lb_cm_old[];
  auto* vertices = reinterpret_cast<VertexMetadata<vid_t,
                                                   metadata_t>*>(
      &shared_lb_cm_old[0]);  // len =  block_size
  auto** nbr_begin = reinterpret_cast<const nbr_t**>(
      &vertices[blockDim.x]);  // len == block size
  auto* row_offset =
      reinterpret_cast<vid_t*>(&nbr_begin[blockDim.x]);  // len = block size
  typedef cub::BlockScan<vid_t, MAX_BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  auto tid = TID_1D;
  auto nthreads = TOTAL_THREADS_1D;
  auto size = work_source.size();
  auto aligned_size = dev::round_up(size, blockDim.x) * blockDim.x;

  assert(blockDim.x <= MAX_BLOCK_SIZE);

  for (uint32_t idx = 0 + tid; idx < aligned_size; idx += nthreads) {
    // calculate valid vertices num in current block
    auto full_tier = aligned_size - blockDim.x;
    auto valid_len = idx < full_tier ? blockDim.x : (size - full_tier);
    vid_t degree = 0;
    assert(valid_len <= blockDim.x);

    if (idx < size) {
      auto v = work_source.GetWork(idx);
      auto adj_list = GetAdjList<FRAG_T, ed>(dev_frag, v);

      vertices[threadIdx.x].set_vertex(v);
      vertices[threadIdx.x].set_metadata(assign_op(v));
      nbr_begin[threadIdx.x] = adj_list.begin_pointer();
      degree = adj_list.Size();
    }

    // Prefix sum
    BlockScan(temp_storage).InclusiveSum(degree, row_offset[threadIdx.x]);

    // make sure the InclusiveSum is finished
    __syncthreads();

    vid_t block_total_edges = row_offset[valid_len - 1];

    for (vid_t eid = threadIdx.x; eid < block_total_edges; eid += blockDim.x) {
      auto v_idx = thrust::upper_bound(thrust::seq, row_offset,
                                       row_offset + valid_len, eid) -
                   row_offset;
      auto offset = eid - (v_idx > 0 ? row_offset[v_idx - 1] : 0);
      auto* nbr = nbr_begin[v_idx] + offset;

      op(vertices[v_idx], *nbr);
    }

    // make sure all threads in the current block are finished
    __syncthreads();
  }
}

template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
          typename EDGE_OP, EdgeDirection ed>
DEV_INLINE void LBCM(const FRAG_T& dev_frag, const WORK_SOURCE_T& work_source,
                     ArrayView<size_t> row_offset, ASSIGN_OP assign_op,
                     EDGE_OP op) {
  using nbr_t = typename FRAG_T::nbr_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;

  extern __shared__ char shared_lb_cm[];
  auto* vertices =
      reinterpret_cast<VertexMetadata<vid_t, metadata_t>*>(&shared_lb_cm[0]);
  auto** nbr_begin = reinterpret_cast<const nbr_t**>(&vertices[blockDim.x]);
  auto* prefix_sum = reinterpret_cast<vid_t*>(&nbr_begin[blockDim.x]);
  size_t size = work_source.size();

  for (size_t block_input_start = blockIdx.x * blockDim.x;
       block_input_start < size; block_input_start += blockDim.x * gridDim.x) {
    size_t block_input_end = min(block_input_start + blockDim.x, size);

    if (block_input_start < block_input_end) {
      auto thread_input = block_input_start + threadIdx.x;
      auto block_output_start =
          (block_input_start >= 1) ? row_offset[block_input_start - 1] : 0;
      auto block_output_end = row_offset[block_input_end - 1];
      auto block_output_size = block_output_end - block_output_start;

      if (thread_input < size) {
        vertex_t u = work_source.GetWork(thread_input);

        vertices[threadIdx.x].set_vertex(u);
        vertices[threadIdx.x].set_metadata(assign_op(u));
        nbr_begin[threadIdx.x] =
            GetAdjList<FRAG_T, ed>(dev_frag, u).begin_pointer();
        prefix_sum[threadIdx.x] = row_offset[thread_input] - block_output_start;
      } else {
        prefix_sum[threadIdx.x] = static_cast<vid_t>(-1);
        nbr_begin[threadIdx.x] = nullptr;
      }

      __syncthreads();

      for (auto eid = threadIdx.x; eid < block_output_size; eid += blockDim.x) {
        auto v_idx = dev::BinarySearch<MAX_BLOCK_SIZE>(prefix_sum, (vid_t) eid);
        auto offset = eid - (v_idx > 0 ? prefix_sum[v_idx - 1] : 0);
        assert(nbr_begin[v_idx] != nullptr);
        auto* nbr = nbr_begin[v_idx] + offset;

        op(vertices[v_idx], *nbr);
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
          typename EDGE_OP, EdgeDirection ed>
DEV_INLINE void LBWARP(const FRAG_T& dev_frag, const WORK_SOURCE_T& work_source,
                       ASSIGN_OP assign_op, EDGE_OP op) {
  using nbr_t = typename FRAG_T::nbr_t;
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;
  const int warp_size = 32;

  extern __shared__ char shared_lb_warp[];
  auto* vertices = reinterpret_cast<VertexMetadata<vid_t,
                                                   metadata_t>*>(
      &shared_lb_warp[0]);  // len =
                            // block_size
  auto** nbr_begin = reinterpret_cast<const nbr_t**>(
      &vertices[blockDim.x]);  // len == block size
  auto* row_offset = reinterpret_cast<vid_t*>(
      &nbr_begin[blockDim.x]);  // len = warp_num * warp_size
  typedef cub::WarpScan<vid_t, warp_size> WarpScan;

  __shared__ typename WarpScan::TempStorage
      temp_storage[MAX_BLOCK_SIZE / warp_size];  // NOLINT(runtime/int)
  auto tid = TID_1D;
  auto warp_id = threadIdx.x / warp_size;
  auto local_tid = threadIdx.x % warp_size;
  auto nthreads = TOTAL_THREADS_1D;
  auto size = work_source.size();
  auto aligned_size = dev::round_up(size, warp_size) * warp_size;

  vertices = &vertices[warp_id * warp_size];
  nbr_begin = &nbr_begin[warp_id * warp_size];
  row_offset = &row_offset[warp_id * warp_size];

  for (uint32_t idx = 0 + tid; idx < aligned_size; idx += nthreads) {
    // calculate valid vertices num in current block
    auto full_tier = aligned_size - warp_size;
    auto valid_len = idx < full_tier ? warp_size : (size - full_tier);

    assert(valid_len <= warp_size);

    vid_t degree = 0;

    if (idx < size) {
      auto v = work_source.GetWork(idx);
      auto adj_list = GetAdjList<FRAG_T, ed>(dev_frag, v);

      vertices[local_tid].set_vertex(v);
      vertices[local_tid].set_metadata(assign_op(v));
      nbr_begin[local_tid] = adj_list.begin_pointer();
      degree = adj_list.Size();
    }

    // Prefix sum
    WarpScan(temp_storage[warp_id]).InclusiveSum(degree, row_offset[local_tid]);

    assert(valid_len > 0);
    vid_t warp_total_edges = row_offset[valid_len - 1];

    for (auto eid = local_tid; eid < warp_total_edges; eid += warp_size) {
      auto v_idx = thrust::upper_bound(thrust::seq, row_offset,
                                       row_offset + valid_len, eid) -
                   row_offset;
      auto offset = eid - ((v_idx > 0) ? row_offset[v_idx - 1] : 0);
      assert(nbr_begin[v_idx] != nullptr);
      auto* nbr = nbr_begin[v_idx] + offset;

      op(vertices[v_idx], *nbr);
    }
    // N.B. Volta needs this for working correctly
    __syncwarp();
  }
}

template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
          typename EDGE_OP, EdgeDirection ed>
DEV_INLINE void LBCTA(const FRAG_T& dev_frag, const WORK_SOURCE_T& work_source,
                      ASSIGN_OP assign_op, EDGE_OP op) {
  using edata_t = typename FRAG_T::edata_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using vid_t = typename FRAG_T::vid_t;

  auto tid = TID_1D;
  auto nthreads = TOTAL_THREADS_1D;
  auto size = work_source.size();
  auto size_rup = dev::round_up(size, blockDim.x) * blockDim.x;
  using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;
  using vertex_metadata_t = VertexMetadata<vid_t, metadata_t>;

  for (auto i = 0 + tid; i < size_rup; i += nthreads) {
    np_local<vid_t, edata_t, vertex_metadata_t> local;

    if (i < size) {
      vertex_metadata_t vm;
      auto v = work_source.GetWork(i);
      auto adj_list = GetAdjList<FRAG_T, ed>(dev_frag, v);

      vm.set_vertex(v);
      vm.set_metadata(assign_op(v));
      local = np_local<vid_t, edata_t, vertex_metadata_t>(adj_list, vm);
    }

    CTAWorkScheduler<vid_t, edata_t, vertex_metadata_t>::schedule(
        local, [=](const Nbr<vid_t, edata_t>& nbr,
                   const vertex_metadata_t& vm) mutable { op(vm, nbr); });
  }
}

template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
          typename EDGE_OP, EdgeDirection ed>
DEV_INLINE void LBSTRICT(const FRAG_T& dev_frag, const ArrayView<size_t>& sidx,
                         const ArrayView<size_t>& row_offset,
                         size_t outputs_per_block,
                         const WORK_SOURCE_T& work_source, ASSIGN_OP assign_op,
                         EDGE_OP op) {
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using nbr_t = typename FRAG_T::nbr_t;
  using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;
  using vertex_metadata_t = VertexMetadata<vid_t, metadata_t>;

  extern __shared__ void* shared_lb_strict[];
  auto* vertices = reinterpret_cast<vertex_metadata_t*>(&shared_lb_strict);
  const nbr_t** nbr_begin =
      reinterpret_cast<const nbr_t**>(&vertices[blockDim.x]);
  size_t* prefix_sum = reinterpret_cast<size_t*>(&nbr_begin[blockDim.x]);
  size_t size = work_source.size();
  size_t total_edges = row_offset[size - 1];
  size_t block_output_start = blockIdx.x * outputs_per_block;

  if (block_output_start >= total_edges) {
    return;
  }

  size_t block_output_end =
      min(block_output_start + outputs_per_block, total_edges);
  size_t block_output_size = block_output_end - block_output_start;
  size_t block_output_processed = 0;

  // the end of index of work_source
  size_t block_input_end =
      (blockIdx.x + 1 == gridDim.x) ? size : min(sidx[blockIdx.x + 1], size);
  // if this block is not last block and the edges of last work block across the
  // current block, then we increase the block_input_end
  if (block_input_end < size &&
      block_output_end >
          (block_input_end > 0 ? row_offset[block_input_end - 1] : 0)) {
    block_input_end++;
  }

  size_t iter_input_start = sidx[blockIdx.x];
  size_t block_first_v_skip_count =
      (block_output_start != row_offset[iter_input_start])
          ? (block_output_start -
             (iter_input_start > 0 ? row_offset[iter_input_start - 1] : 0))
          : 0;

  while (block_output_processed < block_output_size &&
         iter_input_start < block_input_end) {
    size_t iter_input_size =
        min((size_t)(blockDim.x - 1), block_input_end - iter_input_start);
    size_t iter_input_end = iter_input_start + iter_input_size;
    size_t iter_output_end =
        iter_input_end < size ? row_offset[iter_input_end] : total_edges;
    iter_output_end = min(iter_output_end, block_output_end);
    size_t iter_output_size =
        min(iter_output_end - block_output_start, block_output_size);
    iter_output_size -= block_output_processed;
    size_t iter_output_end_offset = iter_output_size + block_output_processed;
    size_t thread_input = iter_input_start + threadIdx.x;

    if (thread_input < block_input_end) {
      auto v = work_source.GetWork(thread_input);
      auto adj_list = GetAdjList<FRAG_T, ed>(dev_frag, v);

      vertices[threadIdx.x].set_vertex(v);
      vertices[threadIdx.x].set_metadata(assign_op(v));
      nbr_begin[threadIdx.x] = adj_list.begin_pointer();
      // filling relative eid among all edge of same source vertex
      prefix_sum[threadIdx.x] = row_offset[thread_input] - block_output_start;
    } else {
      prefix_sum[threadIdx.x] = static_cast<size_t>(-1);
      nbr_begin[threadIdx.x] = nullptr;
    }
    __syncthreads();

    for (vid_t eid = threadIdx.x + block_output_processed;
         eid < iter_output_end_offset; eid += blockDim.x) {
      auto v_idx = dev::BinarySearch<MAX_BLOCK_SIZE>(prefix_sum, (size_t) eid);

      if (v_idx > 0) {
        block_first_v_skip_count = 0;
      }
      auto offset = eid + block_first_v_skip_count -
                    (v_idx > 0 ? prefix_sum[v_idx - 1] : 0);
      auto* nbr = nbr_begin[v_idx] + offset;

      op(vertices[v_idx], *nbr);
    }

    block_output_processed += iter_output_size;
    iter_input_start += iter_input_size;
    block_first_v_skip_count = 0;

    __syncthreads();
  }
}

class ParallelEngine {
 public:
  ParallelEngine() { CHECK_CUDA(cudaProfilerStart()); }

  virtual ~ParallelEngine() { CHECK_CUDA(cudaProfilerStop()); }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdge(const Stream& stream, const FRAG_T& dev_frag,
                          const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                          EDGE_OP op, LoadBalancing lb) {
    if (lb == LoadBalancing::kCMOld) {
      ForEachEdgeCMOld<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          stream, dev_frag, ws, assign_op, op);
    } else if (lb == LoadBalancing::kCM) {
      ForEachEdgeCM<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          stream, dev_frag, ws, assign_op, op);
    } else if (lb == LoadBalancing::kWarp) {
      ForEachEdgeWarp<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          stream, dev_frag, ws, assign_op, op);
    } else if (lb == LoadBalancing::kCTA) {
      ForEachEdgeCTA<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          stream, dev_frag, ws, assign_op, op);
    } else if (lb == LoadBalancing::kStrict) {
      ForEachEdgeStrict<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          stream, dev_frag, ws, assign_op, op);
    } else if (lb == LoadBalancing::kNone) {
      ForEachEdgeNone<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          stream, dev_frag, ws, assign_op, op);
    } else {
      LOG(FATAL) << "Invalid lb policy";
    }
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP>
  inline void ForEachOutgoingEdge(const Stream& stream, const FRAG_T& dev_frag,
                                  const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                                  EDGE_OP op, LoadBalancing lb) {
    ForEachEdge<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP,
                EdgeDirection::kOutgoing>(stream, dev_frag, ws, assign_op, op,
                                          lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename EDGE_OP>
  inline void ForEachOutgoingEdge(const Stream& stream, const FRAG_T& dev_frag,
                                  const WORK_SOURCE_T& ws, EDGE_OP op,
                                  LoadBalancing lb) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;
    using nbr_t = typename FRAG_T::nbr_t;
    ForEachOutgoingEdge(
        stream, dev_frag, ws,
        [] __device__(vertex_t) -> grape::EmptyType { return {}; },
        [=] __device__(
            const VertexMetadata<vid_t, grape::EmptyType>& vertex_metadata,
            const nbr_t& nbr) mutable { op(vertex_metadata.vertex, nbr); },
        lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP>
  inline void ForEachOutgoingInnerVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             ASSIGN_OP assign_op, EDGE_OP op,
                                             LoadBalancing lb) {
    ForEachEdge<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP,
                EdgeDirection::kOutgoingInner>(stream, dev_frag, ws, assign_op,
                                               op, lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename EDGE_OP>
  inline void ForEachOutgoingInnerVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             EDGE_OP op, LoadBalancing lb) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;
    using nbr_t = typename FRAG_T::nbr_t;
    ForEachOutgoingInnerVertexEdge(
        stream, dev_frag, ws,
        [] __device__(vertex_t) -> grape::EmptyType { return {}; },
        [=] __device__(
            const VertexMetadata<vid_t, grape::EmptyType>& vertex_metadata,
            const nbr_t& nbr) mutable { op(vertex_metadata.vertex, nbr); },
        lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP>
  inline void ForEachOutgoingOuterVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             ASSIGN_OP assign_op, EDGE_OP op,
                                             LoadBalancing lb) {
    ForEachEdge<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP,
                EdgeDirection::kOutgoingOuter>(stream, dev_frag, ws, assign_op,
                                               op, lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename EDGE_OP>
  inline void ForEachOutgoingOuterVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             EDGE_OP op, LoadBalancing lb) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;
    using nbr_t = typename FRAG_T::nbr_t;
    ForEachOutgoingOuterVertexEdge(
        stream, dev_frag, ws,
        [] __device__(vertex_t) -> grape::EmptyType { return {}; },
        [=] __device__(
            const VertexMetadata<vid_t, grape::EmptyType>& vertex_metadata,
            const nbr_t& nbr) mutable { op(vertex_metadata.vertex, nbr); },
        lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP>
  inline void ForEachIncomingEdge(const Stream& stream, const FRAG_T& dev_frag,
                                  const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                                  EDGE_OP op, LoadBalancing lb) {
    ForEachEdge<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP,
                EdgeDirection::kIncoming>(stream, dev_frag, ws, assign_op, op,
                                          lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename EDGE_OP>
  inline void ForEachIncomingEdge(const Stream& stream, const FRAG_T& dev_frag,
                                  const WORK_SOURCE_T& ws, EDGE_OP op,
                                  LoadBalancing lb) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;
    using nbr_t = typename FRAG_T::nbr_t;
    ForEachIncomingEdge(
        stream, dev_frag, ws,
        [] __device__(vertex_t) -> grape::EmptyType { return {}; },
        [=] __device__(
            const VertexMetadata<vid_t, grape::EmptyType>& vertex_metadata,
            const nbr_t& nbr) mutable { op(vertex_metadata.vertex, nbr); },
        lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP>
  inline void ForEachIncomingInnerVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             ASSIGN_OP assign_op, EDGE_OP op,
                                             LoadBalancing lb) {
    ForEachEdge<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP,
                EdgeDirection::kIncomingInner>(stream, dev_frag, ws, assign_op,
                                               op, lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename EDGE_OP>
  inline void ForEachIncomingInnerVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             EDGE_OP op, LoadBalancing lb) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;
    using nbr_t = typename FRAG_T::nbr_t;
    ForEachIncomingInnerVertexEdge(
        stream, dev_frag, ws,
        [] __device__(vertex_t) -> grape::EmptyType { return {}; },
        [=] __device__(
            const VertexMetadata<vid_t, grape::EmptyType>& vertex_metadata,
            const nbr_t& nbr) mutable { op(vertex_metadata.vertex, nbr); },
        lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP>
  inline void ForEachIncomingOuterVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             ASSIGN_OP assign_op, EDGE_OP op,
                                             LoadBalancing lb) {
    ForEachEdge<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP,
                EdgeDirection::kIncomingOuter>(stream, dev_frag, ws, assign_op,
                                               op, lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename EDGE_OP>
  inline void ForEachIncomingOuterVertexEdge(const Stream& stream,
                                             const FRAG_T& dev_frag,
                                             const WORK_SOURCE_T& ws,
                                             EDGE_OP op, LoadBalancing lb) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;
    using nbr_t = typename FRAG_T::nbr_t;

    ForEachIncomingOuterVertexEdge(
        stream, dev_frag, ws,
        [] __device__(vertex_t) -> grape::EmptyType { return {}; },
        [=] __device__(
            const VertexMetadata<vid_t, grape::EmptyType>& vertex_metadata,
            const nbr_t& nbr) mutable { op(vertex_metadata.vertex, nbr); },
        lb);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdgeCM(const Stream& stream, const FRAG_T& dev_frag,
                            const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                            EDGE_OP op) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;

    int grid_size, block_size;
    size_t size = ws.size();
    if (size == 0) {
      return;
    }

    __calc_prefix_sum__<FRAG_T, WORK_SOURCE_T, ed>(stream, dev_frag, ws);

    auto calc_shmem_size = [] DEV_HOST(int block_size) -> int {
      using vid_t = typename FRAG_T::vid_t;
      using vertex_t = typename FRAG_T::vertex_t;
      using nbr_t = typename FRAG_T::nbr_t;
      using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;

      return block_size * (sizeof(VertexMetadata<vid_t, metadata_t>) +
                           sizeof(const nbr_t*) + sizeof(vid_t));
    };
    ArrayView<size_t> row_offset_view(
        thrust::raw_pointer_cast(prefix_sum_.data()), size);

    auto lb_wrapper = [=] __device__() {
      LBCM<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          dev_frag, ws, row_offset_view, assign_op, op);
    };  // NOLINT

    KernelSizing(grid_size, block_size, ws.size());
    KernelWrapper<<<grid_size, block_size, calc_shmem_size(block_size),
                    stream.cuda_stream()>>>(lb_wrapper);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdgeCTA(const Stream& stream, const FRAG_T& dev_frag,
                             const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                             EDGE_OP op) {
    int grid_size, block_size;
    auto size = ws.size();
    if (size == 0) {
      return;
    }

    auto lb_wrapper = [=] __device__() {
      LBCTA<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(dev_frag, ws,
                                                           assign_op, op);
    };  // NOLINT

    KernelSizing(grid_size, block_size, size);
    KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
        lb_wrapper);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdgeCMOld(const Stream& stream, const FRAG_T& dev_frag,
                               const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                               EDGE_OP op) {
    int grid_size, block_size;
    auto size = ws.size();
    if (size == 0) {
      return;
    }

    auto calc_shmem_size = [] DEV_HOST(int block_size) -> int {
      using vid_t = typename FRAG_T::vid_t;
      using vertex_t = typename FRAG_T::vertex_t;
      using nbr_t = typename FRAG_T::nbr_t;
      using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;

      return block_size * (sizeof(VertexMetadata<vid_t, metadata_t>) +
                           sizeof(const nbr_t*) + sizeof(vid_t));
    };

    auto lb_wrapper = [=] __device__() {
      LBCMOld<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(dev_frag, ws,
                                                             assign_op, op);
    };  // NOLINT

    KernelSizing(grid_size, block_size, ws.size());
    KernelWrapper<<<grid_size, block_size, calc_shmem_size(block_size),
                    stream.cuda_stream()>>>(lb_wrapper);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdgeWarp(const Stream& stream, const FRAG_T& dev_frag,
                              const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                              EDGE_OP op) {
    int grid_size, block_size;
    auto size = ws.size();
    if (size == 0) {
      return;
    }
    auto calc_shmem_size = [] DEV_HOST(int block_size) -> int {
      using vid_t = typename FRAG_T::vid_t;
      using vertex_t = typename FRAG_T::vertex_t;
      using nbr_t = typename FRAG_T::nbr_t;
      using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;

      int warp_size = 32;
      auto n_warp = (block_size + warp_size - 1) / warp_size;

      auto smem_size = block_size * (sizeof(VertexMetadata<vid_t, metadata_t>) +
                                     sizeof(const nbr_t*)) +
                       n_warp * warp_size * sizeof(vid_t);
      return smem_size;
    };

    auto lb_wrapper = [=] __device__() {
      LBWARP<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(dev_frag, ws,
                                                            assign_op, op);
    };  // NOLINT

    KernelSizing(grid_size, block_size, size);
    KernelWrapper<<<grid_size, block_size, calc_shmem_size(block_size),
                    stream.cuda_stream()>>>(lb_wrapper);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdgeStrict(const Stream& stream, const FRAG_T& dev_frag,
                                const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                                EDGE_OP op) {
    using vid_t = typename FRAG_T::vid_t;
    using vertex_t = typename FRAG_T::vertex_t;

    auto size = ws.size();

    if (size == 0) {
      return;
    }

    __calc_prefix_sum__<FRAG_T, WORK_SOURCE_T, ed>(stream, dev_frag, ws);

    auto lb_wrapper = [=] __device__(const ArrayView<size_t>& sidx,
                                     const ArrayView<size_t>& row_offset,
                                     size_t partition_size) {
      LBSTRICT<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(
          dev_frag, sidx, row_offset, partition_size, ws, assign_op, op);
    };  // NOLINT
    auto calc_shmem_size = [] DEV_HOST(int block_size) -> int {
      using nbr_t = typename FRAG_T::nbr_t;
      using vid_t = typename FRAG_T::vid_t;
      using metadata_t = typename std::result_of<ASSIGN_OP&(vertex_t&)>::type;
      using vertex_metadata_t = VertexMetadata<vid_t, metadata_t>;

      return block_size * (sizeof(const nbr_t*) + sizeof(vertex_metadata_t) +
                           sizeof(size_t));
    };

    size_t total_edges = prefix_sum_[size - 1];

    int block_num, block_size;

    KernelSizing(block_num, block_size, ws.size());

    thrust::device_vector<size_t> sidx(block_num);
    // starting eid in each block
    thrust::device_vector<size_t> seid_per_block(block_num);

    size_t partition_size = dev::round_up(total_edges, block_num);

    LaunchKernel(
        stream,
        [=] __device__(ArrayView<size_t> & starting_eid) {
          auto tid = TID_1D;
          auto nthreads = TOTAL_THREADS_1D;

          for (size_t idx = 0 + tid; idx < block_num; idx += nthreads) {
            starting_eid[idx] = idx * partition_size;
          }
        },
        ArrayView<size_t>(seid_per_block));

    sorted_search<mgpu::bounds_lower>(
        stream, thrust::raw_pointer_cast(seid_per_block.data()), block_num,
        thrust::raw_pointer_cast(prefix_sum_.data()), size,
        thrust::raw_pointer_cast(sidx.data()), mgpu::less_t<size_t>());

    KernelWrapper<<<block_num, block_size, calc_shmem_size(block_size),
                    stream.cuda_stream()>>>(
        lb_wrapper, ArrayView<size_t>(sidx),
        ArrayView<size_t>(thrust::raw_pointer_cast(prefix_sum_.data()), size),
        partition_size);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename ASSIGN_OP,
            typename EDGE_OP, EdgeDirection ed>
  inline void ForEachEdgeNone(const Stream& stream, const FRAG_T& dev_frag,
                              const WORK_SOURCE_T& ws, ASSIGN_OP assign_op,
                              EDGE_OP op) {
    int grid_size, block_size;
    auto size = ws.size();
    if (size == 0) {
      return;
    }
    auto lb_wrapper = [=] __device__() {
      LBNONE<FRAG_T, WORK_SOURCE_T, ASSIGN_OP, EDGE_OP, ed>(dev_frag, ws,
                                                            assign_op, op);
    };  // NOLINT

    KernelSizing(grid_size, block_size, size);
    KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
        lb_wrapper);
  }

  template <typename FRAG_T, typename WORK_SOURCE_T, EdgeDirection ed>
  void __calc_prefix_sum__(const Stream& stream, const FRAG_T& dev_frag,
                           const WORK_SOURCE_T& ws) {
    using vertex_t = typename FRAG_T::vertex_t;
    auto size = ws.size();

    if (size == 0) {
      return;
    }

    if (degree_.size() < size) {
      degree_.resize(size);
    }

    if (prefix_sum_.size() < size) {
      prefix_sum_.resize(size);
    }

    auto* d_degree = thrust::raw_pointer_cast(degree_.data());
    auto* d_prefix_sum = thrust::raw_pointer_cast(prefix_sum_.data());

    ForEachWithIndex(stream, ws, [=] __device__(size_t idx, vertex_t v) {
      d_degree[idx] = GetAdjList<FRAG_T, ed>(dev_frag, v).Size();
    });

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_degree, d_prefix_sum, size,
                                             stream.cuda_stream()));
    CHECK_CUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes,
                               stream.cuda_stream()));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_degree, d_prefix_sum, size,
                                             stream.cuda_stream()));
    CHECK_CUDA(cudaFreeAsync(d_temp_storage, stream.cuda_stream()));
    stream.Sync();
  }

 private:
  // uint32 should be enough for any graph
  thrust::device_vector<uint32_t> degree_;
  thrust::device_vector<size_t> prefix_sum_;
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_PARALLEL_PARALLEL_ENGINE_H_
