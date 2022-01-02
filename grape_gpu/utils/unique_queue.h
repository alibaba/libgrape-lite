#ifndef GRAPE_GPU_UTILS_UNIQUE_QUEUE_H_
#define GRAPE_GPU_UTILS_UNIQUE_QUEUE_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cassert>
#include <cub/util_ptx.cuh>
#include <initializer_list>
#include <memory>
#include <vector>

#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/vertex_set.h"

namespace grape_gpu {
namespace dev {
template <typename VID_T, typename T>
class UniqueQueue {
 public:
  DEV_HOST explicit UniqueQueue(
      const DenseVertexSet<VID_T>& vs,
      const ArrayView<thrust::pair<Vertex<VID_T>, T>>& data, uint64_t* size)
      : vs_(vs), data_(data), size_(size) {}

  DEV_INLINE bool Append(Vertex<VID_T> v, const T& val) {
    if (vs_.Insert(v)) {
      auto allocation = atomicAdd((unsigned long long int*) size_, 1);
      assert(allocation < data_.size());
      data_[allocation] = thrust::make_pair(v, val);
    }
  }

  DEV_INLINE void Clear() const {
    vs_.Clear();
    *size_ = 0;
  }

  DEV_INLINE thrust::pair<Vertex<VID_T>, T>& operator[](uint64_t i) {
    return data_[i];
  }

  DEV_INLINE const thrust::pair<Vertex<VID_T>, T>& operator[](
      uint64_t i) const {
    return data_[i];
  }

  DEV_INLINE uint64_t size() const { return *size_; }

 private:
  DenseVertexSet<VID_T> vs_;
  ArrayView<thrust::pair<Vertex<VID_T>, T>> data_;
  uint64_t* size_;
};
}  // namespace dev

template <typename VID_T, typename T>
class UniqueQueue {
 public:
  void Init(const VertexRange<VID_T>& range) {
    vertex_set_.Init(range);
    data_.resize(range.size());
    size_.set(0);
  }

  void Clear() {
    vertex_set_.Clear();
    size_.set(0);
  }

  void Clear(const Stream& stream) {
    vertex_set_.Clear(stream);
    size_.set(0, stream);
  }

  uint64_t size() const { return size_.get(); }

  uint64_t size(const Stream& stream) const { return size_.get(stream); }

  dev::UniqueQueue<VID_T, T> DeviceObject() {
    return dev::UniqueQueue<VID_T, T>(
        vertex_set_.DeviceObject(),
        ArrayView<thrust::pair<Vertex<VID_T>, T>>(data_), size_.data());
  }

  void Swap(UniqueQueue<VID_T, T>& rhs) {
    vertex_set_.Swap(rhs.vertex_set_);
    data_.swap(rhs.data_);
    size_.Swap(rhs.size_);
  }

 private:
  DenseVertexSet<VID_T> vertex_set_;
  thrust::device_vector<thrust::pair<Vertex<VID_T>, T>> data_;
  SharedValue<uint64_t> size_;
};
}  // namespace grape_gpu

#endif  // GRAPE_GPU_UTILS_UNIQUE_QUEUE_H_