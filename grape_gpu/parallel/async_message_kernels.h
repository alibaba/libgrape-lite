#ifndef GRAPE_GPU_PARALLEL_ASYNC_MESSAGE_KERNELS_H_
#define GRAPE_GPU_PARALLEL_ASYNC_MESSAGE_KERNELS_H_
#include <thrust/pair.h>

#include <cassert>

#include "grape/types.h"
#include "grape_gpu/serialization/async_out_archive.h"

namespace grape_gpu {
namespace kernel {
template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T, typename S>
__global__ typename std::enable_if<
    !std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(dev::AsyncOutArchive<S> recv, const GRAPH_T frag, FUNC_T func) {
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

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T, typename S>
__global__ typename std::enable_if<
    std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(dev::AsyncOutArchive<S> recv, const GRAPH_T frag, FUNC_T func) {
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

template <typename MESSAGE_T, typename FUNC_T, typename S>
__global__ void ProcessMsg(dev::AsyncOutArchive<S> recv, FUNC_T func) {
  MESSAGE_T msg;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(msg);

    if (success) {
      func(msg);
    }
  }
}

}  // namespace kernel
}  // namespace grape_gpu
#endif  // GRAPE_GPU_PARALLEL_ASYNC_MESSAGE_KERNELS_H_
