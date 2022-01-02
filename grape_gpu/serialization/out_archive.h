#ifndef GRAPE_GPU_SERIALIZATION_OUT_ARCHIVE_H_
#define GRAPE_GPU_SERIALIZATION_OUT_ARCHIVE_H_
#include <cooperative_groups.h>

#include "grape/util.h"
#include "grape_gpu/serialization/in_archive.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/device_buffer.h"
#include "grape_gpu/utils/stream.h"

namespace grape_gpu {

namespace dev {

class OutArchive {
 public:
  OutArchive() = default;

  OutArchive(const ArrayView<char>& data, uint32_t* offset)
      : buffer_(data), pos_(offset) {}

  template <typename T>
  DEV_INLINE bool GetBytes(T& elem) {
    auto size = buffer_.size();

    if (size > 0) {
      auto last_offset = atomicAdd(pos_, sizeof(T));
      if (last_offset < size) {
        elem = *reinterpret_cast<T*>(buffer_.data() + last_offset);
        return true;
      }
    }
    return false;
  }

  template <typename T>
  DEV_INLINE bool GetBytesWarp(T& elem) {
    auto size = buffer_.size();

    if (size > 0) {
      auto g = cooperative_groups::coalesced_threads();
      unsigned long long int allocation;

      if (g.thread_rank() == 0) {
        allocation = atomicAdd(pos_, g.size() * sizeof(T));
      }
      auto last_offset = g.shfl(allocation, 0) + g.thread_rank() * sizeof(T);

      if (last_offset < size) {
        elem = *reinterpret_cast<T*>(buffer_.data() + last_offset);
        return true;
      }
    }
    return false;
  }

  DEV_INLINE bool Empty() const {
    auto size = buffer_.size();
    return size == 0 || *pos_ >= size;
  }

  DEV_INLINE uint32_t size() const { return buffer_.size(); }

  DEV_INLINE char* data() { return buffer_.data(); }

 private:
  ArrayView<char> buffer_;
  uint32_t* pos_{};
};

}  // namespace dev

class OutArchive {
 public:
  OutArchive() : OutArchive(0) {}

  explicit OutArchive(uint32_t capacity) : limit_(0), buffer_(capacity) {
    pos_.set(0);
  }

  explicit OutArchive(const InArchive& ia) {
    auto size = ia.size();

    buffer_.resize(size);
    CHECK_CUDA(cudaMemcpy(thrust::raw_pointer_cast(buffer_.data()),
                          thrust::raw_pointer_cast(ia.buffer_.data()), size,
                          cudaMemcpyDeviceToDevice));
    limit_ = size;
    pos_.set(0);
  }

  dev::OutArchive DeviceObject() {
    return {ArrayView<char>(thrust::raw_pointer_cast(buffer_.data()), limit_),
            pos_.data()};
  }

  void Clear() {
    limit_ = 0;
    pos_.set(0);
  }

  void Clear(const Stream& stream) {
    limit_ = 0;
    pos_.set(0, stream);
  }

  void Allocate(uint32_t capacity) { buffer_.resize(capacity); }

  char* data() { return thrust::raw_pointer_cast(buffer_.data()); }

  void SetLimit(uint32_t limit) { limit_ = limit; }

  uint32_t AvailableBytes() const { return limit_ - pos_.get(); }

  uint32_t AvailableBytes(const Stream& stream) const {
    return limit_ - pos_.get(stream);
  }

 private:
  // pos < limit <= buffer_.size()
  thrust::device_vector<char> buffer_;
  uint32_t limit_;
  SharedValue<uint32_t> pos_;
};

class OutArchiveGroup {
 public:
  OutArchiveGroup() = default;

  explicit OutArchiveGroup(size_t group_size)
      : buffers_(group_size), pos_(group_size) {}

  void Init(size_t group_size) {
    buffers_.resize(group_size);
    limits_.resize(group_size, 0);
    pos_.resize(group_size);
    pos_.fill(0);
  }

  dev::OutArchive DeviceObject(size_t idx) {
    return {ArrayView<char>(buffers_[idx].data(), limits_[idx]),
            pos_.data(idx)};
  }

  void Clear(const Stream& stream) {
    limits_.assign(limits_.size(), 0);
    pos_.fill(0, stream);
  }

  void resize(size_t idx, uint32_t size) {
    buffers_[idx].resize(size);
    limits_[idx] = size;
  }

  char* data(size_t idx) { return buffers_[idx].data(); }

  size_t AvailableBytes(size_t idx) const {
    return limits_[idx];
  }

 private:
  std::vector<DeviceBuffer<char>> buffers_;
  std::vector<size_t> limits_;
  SharedArray<uint32_t> pos_;
};
}  // namespace grape_gpu

#endif  // GRAPE_GPU_SERIALIZATION_OUT_ARCHIVE_H_
