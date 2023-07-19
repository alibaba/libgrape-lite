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

#ifndef GRAPE_CUDA_UTILS_ARRAY_VIEW_H_
#define GRAPE_CUDA_UTILS_ARRAY_VIEW_H_

#include "grape/cuda/utils/cuda_utils.h"

namespace grape {
namespace cuda {
template <typename T>
class ArrayView {
 public:
  ArrayView() = default;

  explicit ArrayView(const thrust::device_vector<T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  explicit ArrayView(const pinned_vector<T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  DEV_HOST ArrayView(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE const T* data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_INLINE T& operator[](size_t i) { return data_[i]; }

  DEV_INLINE const T& operator[](size_t i) const { return data_[i]; }

  DEV_INLINE void Swap(ArrayView<T>& rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_INLINE T* begin() { return data_; }

  DEV_INLINE T* end() { return data_ + size_; }

  DEV_INLINE const T* begin() const { return data_; }

  DEV_INLINE const T* end() const { return data_ + size_; }

 private:
  T* data_{};
  size_t size_{};
};

}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_ARRAY_VIEW_H_
