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

#ifndef GRAPE_CUDA_UTILS_VERTEX_ARRAY_H_
#define GRAPE_CUDA_UTILS_VERTEX_ARRAY_H_

#pragma push
#pragma diag_suppress = initialization_not_reachable
#include <thrust/device_vector.h>

#include <cub/util_type.cuh>
#pragma pop

#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/stream.h"
#include "grape/utils/gcontainer.h"
#include "grape/utils/vertex_array.h"

namespace grape {
namespace cuda {
namespace dev {
template <typename T, typename VID_T>
class VertexArray {
 public:
  VertexArray() = default;

  DEV_HOST VertexArray(VertexRange<VID_T> range, T* data)
      : range_(range), data_(data), fake_start_(data - range.begin_value()) {}

  DEV_INLINE T& operator[](const Vertex<VID_T>& loc) {
    return fake_start_[loc.GetValue()];
  }

  DEV_INLINE const T& operator[](const Vertex<VID_T>& loc) const {
    return fake_start_[loc.GetValue()];
  }

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE size_t size() const { return range_.size(); }

 private:
  VertexRange<VID_T> range_;
  T* data_{};
  T* fake_start_{};
};

}  // namespace dev

template <typename T, typename VID_T>
class VertexArray : public grape::Array<T, grape::Allocator<T>> {
  using Base = grape::Array<T, grape::Allocator<T>>;

 public:
  VertexArray() : Base(), fake_start_(NULL) {}

  explicit VertexArray(const VertexRange<VID_T>& range)
      : Base(range.size()), range_(range) {
    fake_start_ = Base::data() - range_.begin_value();
    d_data_.resize(range.size());
  }

  VertexArray(const VertexRange<VID_T>& range, const T& value)
      : Base(range.size(), value), range_(range) {
    fake_start_ = Base::data() - range_.begin_value();
    d_data_.resize(range.size());
  }

  ~VertexArray() = default;

  void Init(const VertexRange<VID_T>& range) {
    Base::clear();
    Base::resize(range.size());
    range_ = range;
    fake_start_ = Base::data() - range_.begin_value();
    d_data_.clear();
    d_data_.resize(range.size());
  }

  void Init(const VertexRange<VID_T>& range, const T& value) {
    Base::clear();
    Base::resize(range.size(), value);
    range_ = range;
    fake_start_ = Base::data() - range_.begin_value();
    d_data_.clear();
    d_data_.resize(range.size(), value);
  }

  void SetValue(VertexRange<VID_T>& range, const T& value) {
    std::fill_n(&Base::data()[range.begin().GetValue() - range_.begin_value()],
                range.size(), value);
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](Vertex<VID_T>& loc) {
    return fake_start_[loc.GetValue()];
  }

  inline const T& operator[](const Vertex<VID_T>& loc) const {
    return fake_start_[loc.GetValue()];
  }

  void resize(size_t size) {
    grape::Array<T, grape::Allocator<T>>::resize(size);
    d_data_.resize(size);
    d_data_.shrink_to_fit();
  }

  void Swap(VertexArray& rhs) {
    Base::swap((Base&) rhs);
    range_.Swap(rhs.range_);
    std::swap(fake_start_, rhs.fake_start_);
    d_data_.swap(rhs.d_data_);
  }

  void Clear() {
    VertexArray ga;
    this->Swap(ga);
  }

  const VertexRange<VID_T>& GetVertexRange() const { return range_; }

  dev::VertexArray<T, VID_T> DeviceObject() {
    return dev::VertexArray<T, VID_T>(range_,
                                      thrust::raw_pointer_cast(d_data_.data()));
  }

  void H2D() {
    CHECK_CUDA(cudaMemcpy(d_data_.data().get(), this->data(),
                          sizeof(T) * this->size(), cudaMemcpyHostToDevice));
  }

  void H2D(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(d_data_.data().get(), this->data(),
                               sizeof(T) * this->size(), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  void D2H() {
    CHECK_CUDA(cudaMemcpy(this->data(), d_data_.data().get(),
                          sizeof(T) * this->size(), cudaMemcpyDeviceToHost));
  }

  void D2H(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(this->data(), d_data_.data().get(),
                               sizeof(T) * this->size(), cudaMemcpyDeviceToHost,
                               stream.cuda_stream()));
  }

 private:
  VertexRange<VID_T> range_;
  T* fake_start_;
  thrust::device_vector<T> d_data_;
};

}  // namespace cuda

}  // namespace grape

namespace cub {
// Making Vertex to be comparable for cub's segmented radix sort
template <typename T>
struct Traits<grape::Vertex<T>>
    : NumericTraits<typename std::remove_cv<T>::Type> {};

}  // namespace cub

#endif  // GRAPE_CUDA_UTILS_VERTEX_ARRAY_H_
