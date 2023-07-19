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

#ifndef GRAPE_CUDA_UTILS_SHARED_VALUE_H_
#define GRAPE_CUDA_UTILS_SHARED_VALUE_H_

#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/stream.h"

namespace grape {
namespace cuda {
template <typename T>
class SharedValue {
  static_assert(std::is_pod<T>::value, "Unsupported datatype");

 public:
  SharedValue() {
    d_buffer_.resize(1);
    h_buffer_.resize(1);
  }

  void set(const T& t) { d_buffer_[0] = t; }

  void set(const T& t, const Stream& stream) {
    h_buffer_[0] = t;
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  typename thrust::device_vector<T>::reference get() { return d_buffer_[0]; }

  typename thrust::device_vector<T>::const_reference get() const {
    return d_buffer_[0];
  }

  T get(const Stream& stream) const {
    CHECK_CUDA(cudaMemcpyAsync(
        const_cast<T*>(thrust::raw_pointer_cast(h_buffer_.data())),
        thrust::raw_pointer_cast(d_buffer_.data()), sizeof(T),
        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_[0];
  }

  T* data() { return thrust::raw_pointer_cast(d_buffer_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(d_buffer_.data()); }

  void Assign(const SharedValue<T>& rhs) {
    CHECK_CUDA(cudaMemcpy(thrust::raw_pointer_cast(d_buffer_.data()),
                          thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Assign(const SharedValue<T>& rhs, const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                               sizeof(T), cudaMemcpyDefault,
                               stream.cuda_stream()));
  }

  void Swap(SharedValue<T>& rhs) { d_buffer_.swap(rhs.d_buffer_); }

 private:
  thrust::device_vector<T> d_buffer_;
  pinned_vector<T> h_buffer_;
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_SHARED_VALUE_H_
