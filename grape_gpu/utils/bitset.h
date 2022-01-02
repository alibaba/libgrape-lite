#ifndef GRAPE_GPU_UTILS_BITSET_H_
#define GRAPE_GPU_UTILS_BITSET_H_
#include <cooperative_groups.h>
#include <thrust/device_vector.h>

#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/vertex_array.h"

namespace grape_gpu {
namespace dev {
template <typename SIZE_T>
class Bitset;

template <>
class Bitset<uint32_t> {
 public:
  __host__ __device__ Bitset(ArrayView<uint64_t> data, uint32_t size,
                             uint32_t* positive_count)
      : data_(data), size_(size), positive_count_(positive_count) {}

  __device__ __forceinline__ bool set_bit(uint32_t pos) {
    assert(pos < size_);
    auto bit = (uint64_t) 1l << bit_offset(pos);
    if (data_[word_offset(pos)] & bit) {
      return false;
    }
    atomicAdd(positive_count_, 1);
    data_[word_offset(pos)] |= bit;
    return true;
  }

  __device__ __forceinline__ bool set_bit_atomic(uint32_t pos) {
    assert(pos < size_);
    uint64_t old_val, new_val;
    do {
      old_val = data_[word_offset(pos)];
      if (old_val & ((uint64_t) 1l << bit_offset(pos))) {
        return false;
      }
      new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
    } while (old_val != atomicCAS(reinterpret_cast<unsigned long long int*>(
                                      data_.data() + word_offset(pos)),
                                  old_val, new_val));
    if ((old_val & (1l << bit_offset(pos))) == 0) {
      auto g = cooperative_groups::coalesced_threads();

      if (g.thread_rank() == 0) {
        atomicAdd(positive_count_, g.size());
      }
      return true;
    }
    return false;
  }

  __device__ __forceinline__ void clear() {
    auto tid = TID_1D;
    auto nthreads = TOTAL_THREADS_1D;

    for (uint64_t i = 0 + tid; i < data_.size(); i += nthreads) {
      data_[i] = 0;
    }

    if (tid == 0) {
      *positive_count_ = 0;
    }
  }

  __device__ __forceinline__ bool get_bit(uint32_t pos) const {
    assert(pos < size_);
    return (data_[word_offset(pos)] >> bit_offset(pos)) & 1l;
  }

  __device__ __forceinline__ uint32_t get_size() const { return data_.size(); }

  __device__ __forceinline__ uint32_t get_positive_count() const {
    return *positive_count_;
  }

 private:
  __device__ __forceinline__ uint64_t word_offset(uint32_t n) const {
    return n / kBitsPerWord;
  }

  __device__ __forceinline__ uint64_t bit_offset(uint32_t n) const {
    return n & (kBitsPerWord - 1);
  }
  static const uint32_t kBitsPerWord = 64;

  ArrayView<uint64_t> data_;
  uint32_t size_{};
  uint32_t* positive_count_{};
};

template <>
class Bitset<uint64_t> {
 public:
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long int));

  __host__ __device__ Bitset(ArrayView<uint64_t> data, uint64_t size,
                             uint64_t* positive_count)
      : data_(data), size_(size), positive_count_(positive_count) {}

  __device__ __forceinline__ bool set_bit(uint64_t pos) {
    assert(pos < size_);
    auto bit = (uint64_t) 1l << bit_offset(pos);
    if (data_[word_offset(pos)] & bit) {
      return false;
    }
    atomicAdd((unsigned long long int*) positive_count_, 1);
    data_[word_offset(pos)] |= bit;
    return true;
  }

  __device__ __forceinline__ bool set_bit_atomic(uint64_t pos) {
    assert(pos < size_);
    uint64_t old_val, new_val;
    do {
      old_val = data_[word_offset(pos)];
      if (old_val & ((uint64_t) 1l << bit_offset(pos))) {
        return false;
      }
      new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
    } while (old_val != atomicCAS(reinterpret_cast<unsigned long long int*>(
                                      data_.data() + word_offset(pos)),
                                  old_val, new_val));
    if ((old_val & (1l << bit_offset(pos))) == 0) {
      auto g = cooperative_groups::coalesced_threads();

      if (g.thread_rank() == 0) {
        atomicAdd((unsigned long long int*) positive_count_, g.size());
      }
      return true;
    }
    return false;
  }

  __device__ __forceinline__ void clear() {
    auto tid = TID_1D;
    auto nthreads = TOTAL_THREADS_1D;

    for (uint64_t i = 0 + tid; i < data_.size(); i += nthreads) {
      data_[i] = 0;
    }

    if (tid == 0) {
      *positive_count_ = 0;
    }
  }

  __device__ __forceinline__ bool get_bit(uint64_t pos) const {
    assert(pos < size_);
    return (data_[word_offset(pos)] >> bit_offset(pos)) & 1l;
  }

  __device__ __forceinline__ uint64_t get_size() const { return data_.size(); }

  __device__ __forceinline__ uint64_t get_positive_count() const {
    return *positive_count_;
  }

 private:
  __device__ __forceinline__ uint64_t word_offset(uint64_t n) const {
    return n / kBitsPerWord;
  }

  __device__ __forceinline__ uint64_t bit_offset(uint64_t n) const {
    return n & (kBitsPerWord - 1);
  }
  static const uint32_t kBitsPerWord = 64;

  ArrayView<uint64_t> data_;
  uint64_t size_{};
  uint64_t* positive_count_{};
};

}  // namespace dev

namespace bitset_kernels {
template <typename SIZE_T>
__global__ void SetBit(dev::Bitset<SIZE_T> bitset, size_t pos) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    bitset.set_bit(pos);
}
}  // namespace bitset_kernels

template <typename SIZE_T>
class Bitset {
 public:
  Bitset() = default;

  void Init(SIZE_T size) {
    size_ = size;
    data_.resize(getNumWords(size), 0);
    positive_count_.set(0);
  }

  dev::Bitset<SIZE_T> DeviceObject() {
    return dev::Bitset<SIZE_T>(grape_gpu::ArrayView<uint64_t>(data_), size_,
                               positive_count_.data());
  }

  void Clear() {
    positive_count_.set(0);
    thrust::fill(data_.begin(), data_.end(), 0);
  }

  void Clear(const Stream& stream) {
    positive_count_.set(0, stream);
    CHECK_CUDA(cudaMemsetAsync(thrust::raw_pointer_cast(data_.data()), 0,
                               sizeof(uint64_t) * data_.size(),
                               stream.cuda_stream()));
  }

  void SetBit(SIZE_T pos) {
    CHECK_LT(pos, size_);
    bitset_kernels::SetBit<SIZE_T><<<1, 1>>>(this->DeviceObject(), pos);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  void SetBit(SIZE_T pos, const Stream& stream) {
    bitset_kernels::SetBit<SIZE_T>
        <<<1, 1, 0, stream.cuda_stream()>>>(this->DeviceObject(), pos);
  }

  void Swap(Bitset<SIZE_T>& other) {
    data_.swap(other.data_);
    std::swap(size_, other.size_);
    positive_count_.Swap(other.positive_count_);
  }

  SIZE_T GetSize() const { return size_; }

  SIZE_T GetPositiveCount() const { return positive_count_.get(); }

  SIZE_T GetPositiveCount(const Stream& stream) const {
    return positive_count_.get(stream);
  }

 private:
  static const uint64_t kBitsPerWord = 64;

  static SIZE_T getNumWords(SIZE_T size) {
    return (size + kBitsPerWord - 1) / kBitsPerWord;
  }

  thrust::device_vector<uint64_t> data_;
  SIZE_T size_{};
  SharedValue<SIZE_T> positive_count_;
};

template <typename SIZE_T>
class RemoteBitset {
 public:
  RemoteBitset() = default;

  void Init(const grape::CommSpec& comm_spec, SIZE_T size) {
    size_ = size;
    data_ = std::move(IPCArray<uint64_t, IPCMemoryPlacement::kDevice>(comm_spec));
    data_.Init(size_);
    positive_count_ = std::move(RemoteSharedValue<SIZE_T>(comm_spec));
    positive_count_.Init();
    positive_count_.set(0);
  }

  dev::Bitset<SIZE_T> DeviceObject() {
    return dev::Bitset<SIZE_T>(data_.local_view(), size_,
                               positive_count_.data());
  }

  dev::Bitset<SIZE_T> DeviceObject(int rid) {
    return dev::Bitset<SIZE_T>(data_.view(rid), size_, positive_count_.data());
  }

  void Clear() {
    positive_count_.set(0);
    data_.Clear();
  }

  void SetBit(SIZE_T pos) {
    CHECK_LT(pos, size_);
    bitset_kernels::SetBit<SIZE_T><<<1, 1>>>(this->DeviceObject(), pos);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  void SetBit(SIZE_T pos, const Stream& stream) {
    bitset_kernels::SetBit<SIZE_T>
        <<<1, 1, 0, stream.cuda_stream()>>>(this->DeviceObject(), pos);
  }

  void Swap(RemoteBitset<SIZE_T>& other) {
    data_.Swap(other.data_);
    std::swap(size_, other.size_);
    positive_count_.Swap(other.positive_count_);
  }

  SIZE_T GetSize() const { return size_; }

  SIZE_T GetPositiveCount() const { return positive_count_.get(); }

  SIZE_T GetPositiveCount(const Stream& stream) const {
    return positive_count_.get(stream);
  }

 private:
  static const uint64_t kBitsPerWord = 64;

  static SIZE_T getNumWords(SIZE_T size) {
    return (size + kBitsPerWord - 1) / kBitsPerWord;
  }

  // thrust::device_vector<uint64_t> data_;
  IPCArray<uint64_t, IPCMemoryPlacement::kDevice> data_;
  SIZE_T size_{};
  RemoteSharedValue<SIZE_T> positive_count_;
};

}  // namespace grape_gpu

#endif  // GRAPE_GPU_UTILS_BITSET_H_
