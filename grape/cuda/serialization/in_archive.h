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

#ifndef GRAPE_CUDA_SERIALIZATION_IN_ARCHIVE_H_
#define GRAPE_CUDA_SERIALIZATION_IN_ARCHIVE_H_

#pragma push
#pragma diag_suppress = initialization_not_reachable
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <cub/cub.cuh>
#pragma pop

#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/dev_utils.h"
#include "grape/cuda/utils/shared_array.h"
#include "grape/cuda/utils/stream.h"

namespace grape {
namespace cuda {
namespace dev {
class InArchive {
 public:
  InArchive() = default;

  InArchive(const ArrayView<char>& buffer, uint32_t* size)
      : buffer_(buffer), size_(size) {}

  template <typename T>
  DEV_INLINE void AddBytes(const T& elem) {
    auto begin = atomicAdd(size_, sizeof(T));

    assert(begin < buffer_.size());
    *reinterpret_cast<T*>(buffer_.data() + begin) = elem;
  }

  template <typename T>
  DEV_INLINE void AddBytesWarpOpt(int fid, const T& elem) {
    auto g = cooperative_groups::coalesced_threads();
    uint32_t mask = g.match_any(fid);
    uint32_t prefix = (1 << g.thread_rank()) - 1;
    uint32_t prefix_num = __popc(mask & prefix);
    uint32_t count = __popc(mask);
    uint32_t leader = __ffs(mask) - 1;
    int warp_res;
    if (g.thread_rank() == leader) {
      warp_res = atomicAdd(size_, count * sizeof(T));
    }
    auto begin = g.shfl(warp_res, leader) + prefix_num * sizeof(T);

    assert(begin < buffer_.size());
    *reinterpret_cast<T*>(buffer_.data() + begin) = elem;
  }

  template <typename T>
  DEV_INLINE void AddBytesWarp(const T& elem) {
    auto g = cooperative_groups::coalesced_threads();
    int warp_res;
    if (g.thread_rank() == 0) {
      warp_res = atomicAdd(size_, g.size() * sizeof(T));
    }
    auto begin = g.shfl(warp_res, 0) + g.thread_rank() * sizeof(T);

    assert(begin < buffer_.size());
    *reinterpret_cast<T*>(buffer_.data() + begin) = elem;
  }

 private:
  ArrayView<char> buffer_;
  uint32_t* size_{};
};

}  // namespace dev

class OutArchive;

class InArchive {
 public:
  InArchive() : InArchive(0) {}

  explicit InArchive(uint32_t capacity) : buffer_(capacity) { size_.set(0); }

  dev::InArchive DeviceObject() {
    return {ArrayView<char>(buffer_), size_.data()};
  }

  void Allocate(uint32_t capacity) { buffer_.resize(capacity); }

  uint32_t size() const { return size_.get(); }

  uint32_t size(const Stream& stream) const { return size_.get(stream); }

  uint32_t capacity() const { return buffer_.size(); }

  void Clear() { size_.set(0); }

  void Clear(const Stream& stream) { size_.set(0, stream); }

  bool Empty() const { return size_.get() == 0; }

  bool Empty(const Stream& stream) const { return size_.get(stream) == 0; }

  char* data() { return thrust::raw_pointer_cast(buffer_.data()); }

 private:
  thrust::device_vector<char> buffer_;
  SharedValue<uint32_t> size_;
  friend class grape::cuda::OutArchive;
};

class InArchiveGroup {
 public:
  InArchiveGroup() = default;

  explicit InArchiveGroup(size_t group_size)
      : buffers_(group_size), sizes_(group_size) {}

  void Init(size_t size) {
    buffers_.resize(size);
    sizes_.resize(size);
  }

  dev::InArchive DeviceObject(size_t idx) {
    return {ArrayView<char>(buffers_[idx]), sizes_.data(idx)};
  }

  void resize(size_t idx, uint32_t capacity) {
    buffers_[idx].resize(capacity);
    buffers_[idx].shrink_to_fit();
  }

  const typename SharedArray<uint32_t>::host_t& size(
      const Stream& stream) const {
    return sizes_.get(stream);
  }

  typename SharedArray<uint32_t>::host_t& size(const Stream& stream) {
    return sizes_.get(stream);
  }

  void Clear(const Stream& stream) { sizes_.fill(0, stream); }

  bool Empty(size_t idx, const Stream& stream) const {
    return sizes_.get(idx, stream) == 0;
  }

  char* data(size_t idx) {
    return thrust::raw_pointer_cast(buffers_[idx].data());
  }

 private:
  std::vector<thrust::device_vector<char>> buffers_;
  SharedArray<uint32_t> sizes_;
  friend class grape::cuda::OutArchive;
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_SERIALIZATION_IN_ARCHIVE_H_
