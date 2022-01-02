#ifndef GRAPE_GPU_SERIALIZATION_IN_ARCHIVE_H_
#define GRAPE_GPU_SERIALIZATION_IN_ARCHIVE_H_
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <cub/cub.cuh>

#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/shared_array.h"
#include "grape_gpu/utils/stream.h"

namespace grape_gpu {
namespace dev {
class InArchive {
 public:
  InArchive() = default;

  InArchive(const ArrayView<char>& buffer, uint32_t* size)
      : buffer_(buffer), size_(size) {}

  template <typename T>
  DEV_INLINE void AddBytes(size_t begin, const T& elem) {
    assert(begin < buffer_.size());
    *reinterpret_cast<T*>(buffer_.data() + begin) = elem;
  }

  template <typename T>
  DEV_INLINE void AddBytes(const T& elem) {
    auto begin = atomicAdd(size_, sizeof(T));

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
  friend class grape_gpu::OutArchive;
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

  void resize(size_t idx, uint32_t capacity) { buffers_[idx].resize(capacity); }

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

  uint32_t* get_size_ptr() { return sizes_.data(); }

 private:
  std::vector<thrust::device_vector<char>> buffers_;
  SharedArray<uint32_t> sizes_;
  friend class grape_gpu::OutArchive;
};

}  // namespace grape_gpu

#endif  // GRAPE_GPU_SERIALIZATION_IN_ARCHIVE_H_
