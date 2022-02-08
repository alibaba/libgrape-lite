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

#ifndef GRAPE_CUDA_UTILS_DEVICE_BUFFER_H_
#define GRAPE_CUDA_UTILS_DEVICE_BUFFER_H_

#include <cuda.h>

#include <memory>

#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/cuda_utils.h"

namespace grape {
namespace cuda {
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t size) { resize(size); }

  DeviceBuffer(const DeviceBuffer<T>& rhs) { *this = rhs; }

  DeviceBuffer(DeviceBuffer<T>&& rhs) noexcept { *this = rhs; }

  ~DeviceBuffer() { CHECK_CUDA(cudaFree(data_)); }

  DeviceBuffer& operator=(const DeviceBuffer<T>& rhs) {
    if (&rhs != this) {
      resize(rhs.size_);
      CHECK_CUDA(cudaMemcpy(data_, rhs.data_, sizeof(T) * rhs.size_,
                            cudaMemcpyDeviceToDevice));
    }
    return *this;
  }

  DeviceBuffer& operator=(DeviceBuffer<T>&& rhs) noexcept {
    if (&rhs != this) {
      CHECK_CUDA(cudaFree(data_));
      data_ = rhs.data_;
      size_ = rhs.size_;
      capacity_ = rhs.capacity_;
      rhs.data_ = nullptr;
      rhs.size_ = 0;
      rhs.capacity_ = 0;
    }
    return *this;
  }

  void resize(size_t size) {
    size_ = size;
    if (size > capacity_) {
      CHECK_CUDA(cudaFree(data_));
      CHECK_CUDA(cudaMalloc(&data_, sizeof(T) * size_));
      capacity_ = size;
    }
  }

  T* data() { return data_; }

  const T* data() const { return data_; }

  size_t size() const { return size_; }

  ArrayView<T> DeviceObject() { return ArrayView<T>(data_, size_); }

 private:
  size_t capacity_{};
  size_t size_{};
  T* data_{};
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_DEVICE_BUFFER_H_
