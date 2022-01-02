#ifndef GRAPE_GPU_PARALLEL_MESSAGE_KERNELS_H_
#define GRAPE_GPU_PARALLEL_MESSAGE_KERNELS_H_
#include <thrust/pair.h>

#include <cassert>

#include "grape/types.h"
#include "grape_gpu/serialization/out_archive.h"

namespace grape_gpu {
namespace kernel {
template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    !std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsgFused(ArrayView<dev::OutArchive> recvs, const GRAPH_T frag,
                FUNC_T func) {
  using unit_t = thrust::pair<typename GRAPH_T::vid_t, MESSAGE_T>;

  for (auto& recv : recvs) {
    auto* data = recv.data();
    auto size_in_bytes = recv.size();
    auto size = round_up(size_in_bytes, sizeof(unit_t));

    assert(size_in_bytes % sizeof(unit_t) == 0);

    for (size_t idx = TID_1D; idx < size; idx += TOTAL_THREADS_1D) {
      auto char_begin = idx * sizeof(unit_t);

      if (char_begin < size_in_bytes) {
        auto& pair = *reinterpret_cast<unit_t*>(data + char_begin);

        typename GRAPH_T::vertex_t v;
        bool success = frag.Gid2Vertex(pair.first, v);
        assert(success);
        func(v, pair.second);
      }
    }
  }
}

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsgFused(ArrayView<dev::OutArchive> recvs, const GRAPH_T frag,
                FUNC_T func) {
  using unit_t = typename GRAPH_T::vid_t;

  for (auto& recv : recvs) {
    auto* data = recv.data();
    auto size_in_bytes = recv.size();
    auto size = round_up(size_in_bytes, sizeof(unit_t));

    for (size_t idx = TID_1D; idx < size; idx += TOTAL_THREADS_1D) {
      auto char_begin = idx * sizeof(unit_t);

      if (char_begin < size_in_bytes) {
        auto& gid = *reinterpret_cast<unit_t*>(data + char_begin);

        typename GRAPH_T::vertex_t v;
        bool success = frag.Gid2Vertex(gid, v);
        assert(success);
        func(v);
      }
    }
  }
}

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    !std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(dev::OutArchive recv, const GRAPH_T frag, FUNC_T func) {
  thrust::pair<typename GRAPH_T::vid_t, MESSAGE_T> pair;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(pair);

    if (success) {
      typename GRAPH_T::vertex_t v;
      bool success = frag.Gid2Vertex(pair.first, v);
      assert(success);
      func(v, pair.second);
    }
  }
}

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(dev::OutArchive recv, const GRAPH_T frag, FUNC_T func) {
  typename GRAPH_T::vid_t gid;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(gid);

    if (success) {
      typename GRAPH_T::vertex_t v;
      bool success = frag.Gid2Vertex(gid, v);
      assert(success);
      func(v);
    }
  }
}

template <typename MESSAGE_T, typename FUNC_T>
__global__ void ProcessMsgFused(ArrayView<dev::OutArchive> recvs, FUNC_T func) {
  for (auto& recv : recvs) {
    auto* data = recv.data();
    auto size_in_bytes = recv.size();
    auto size = round_up(size_in_bytes, sizeof(MESSAGE_T));

    for (size_t idx = TID_1D; idx < size; idx += TOTAL_THREADS_1D) {
      auto char_begin = idx * sizeof(MESSAGE_T);

      if (char_begin < size_in_bytes) {
        func(*reinterpret_cast<MESSAGE_T*>(data + char_begin));
      }
    }
  }
}

}  // namespace kernel
}  // namespace grape_gpu
#endif  // GRAPE_GPU_PARALLEL_MESSAGE_KERNELS_H_
