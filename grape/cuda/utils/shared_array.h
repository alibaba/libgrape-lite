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

#ifndef GRAPE_CUDA_UTILS_SHARED_ARRAY_H_
#define GRAPE_CUDA_UTILS_SHARED_ARRAY_H_

#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/stream.h"

namespace grape {
namespace cuda {
template <typename T>
class SharedArray {
  static_assert(std::is_pod<T>::value, "Unsupported datatype");

 public:
  using device_t = thrust::device_vector<T>;
  using host_t = pinned_vector<T>;

  SharedArray() = default;

  explicit SharedArray(size_t size) {
    d_buffer_.resize(size);
    h_buffer_.resize(size);
  }

  void resize(size_t size) {
    d_buffer_.resize(size);
    h_buffer_.resize(size);
  }

  void set(size_t idx, const T& t) { d_buffer_[idx] = t; }

  void set(size_t idx, const T& t, const Stream& stream) {
    h_buffer_[idx] = t;
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  void fill(const T& t) {
    auto size = h_buffer_.size();
    thrust::fill_n(h_buffer_.data(), size, t);
    d_buffer_ = h_buffer_;
  }

  void fill(const T& t, const Stream& stream) {
    auto size = h_buffer_.size();

    thrust::fill_n(h_buffer_.data(), size, t);
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T) * size, cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  typename thrust::device_vector<T>::reference get(size_t idx) {
    return d_buffer_[idx];
  }

  typename thrust::device_vector<T>::const_reference get(size_t idx) const {
    return d_buffer_[idx];
  }

  T get(size_t idx, const Stream& stream) const {
    CHECK_CUDA(cudaMemcpyAsync(
        const_cast<T*>((thrust::raw_pointer_cast(h_buffer_.data()) + idx)),
        thrust::raw_pointer_cast(d_buffer_.data()) + idx, sizeof(T),
        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_[idx];
  }

  const host_t& get(const Stream& stream) const {
    CHECK_CUDA(cudaMemcpyAsync(
        const_cast<T*>(thrust::raw_pointer_cast(h_buffer_.data())),
        thrust::raw_pointer_cast(d_buffer_.data()),
        sizeof(T) * d_buffer_.size(), cudaMemcpyDeviceToHost,
        stream.cuda_stream()));
    stream.Sync();
    return h_buffer_;
  }

  host_t& get(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(h_buffer_.data()),
                               thrust::raw_pointer_cast(d_buffer_.data()),
                               sizeof(T) * d_buffer_.size(),
                               cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_;
  }

  T* data() { return thrust::raw_pointer_cast(d_buffer_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(d_buffer_.data()); }

  T* data(size_t idx) {
    return thrust::raw_pointer_cast(d_buffer_.data()) + idx;
  }

  const T* data(size_t idx) const {
    return thrust::raw_pointer_cast(d_buffer_.data()) + idx;
  }

  void Assign(const SharedArray<T>& rhs) {
    CHECK_CUDA(cudaMemcpy(thrust::raw_pointer_cast(d_buffer_.data()),
                          thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Assign(const SharedArray<T>& rhs, const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                               sizeof(T), cudaMemcpyDefault,
                               stream.cuda_stream()));
  }

  void Swap(SharedArray<T>& rhs) { d_buffer_.swap(rhs.d_buffer_); }

 private:
  device_t d_buffer_;
  host_t h_buffer_;
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_SHARED_ARRAY_H_
