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

#ifndef GRAPE_CUDA_PARALLEL_MESSAGE_KERNELS_H_
#define GRAPE_CUDA_PARALLEL_MESSAGE_KERNELS_H_
#include <thrust/pair.h>

#include <cassert>

#include "grape/cuda/serialization/out_archive.h"
#include "grape/types.h"

namespace grape {
namespace cuda {
namespace dev {
template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    !std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(ArrayView<dev::OutArchive> recvs, const GRAPH_T frag, FUNC_T func) {
  using unit_t = thrust::pair<typename GRAPH_T::vid_t, MESSAGE_T>;

  for (auto& recv : recvs) {
    auto* data = recv.data();
    auto size_in_bytes = recv.size();
    auto size = round_up(size_in_bytes, sizeof(unit_t));

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

template <typename GRAPH_T, typename MESSAGE_T, typename FUNC_T>
__global__ typename std::enable_if<
    std::is_same<MESSAGE_T, grape::EmptyType>::value>::type
ProcessMsg(ArrayView<dev::OutArchive> recvs, const GRAPH_T frag, FUNC_T func) {
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

template <typename MESSAGE_T, typename FUNC_T>
__global__ void ProcessMsg(dev::OutArchive recv, FUNC_T func) {
  MESSAGE_T msg;

  while (!recv.Empty()) {
    bool success = recv.GetBytesWarp(msg);

    if (success) {
      func(msg);
    }
  }
}

}  // namespace dev
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_PARALLEL_MESSAGE_KERNELS_H_
