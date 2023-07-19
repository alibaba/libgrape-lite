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

#ifndef GRAPE_CUDA_UTILS_WORK_SOURCE_H_
#define GRAPE_CUDA_UTILS_WORK_SOURCE_H_
#include "grape/config.h"
#include "grape/cuda/utils/cuda_utils.h"
namespace grape {
namespace cuda {
template <typename T>
struct WorkSourceRange {
 public:
  DEV_HOST WorkSourceRange(T start, size_t size) : start_(start), size_(size) {}

  DEV_HOST_INLINE T GetWork(size_t i) const { return (T) (start_ + i); }

  DEV_HOST_INLINE size_t size() const { return size_; }

 private:
  T start_;
  size_t size_;
};

template <typename T>
struct WorkSourceArray {
 public:
  DEV_HOST WorkSourceArray(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T GetWork(size_t i) const { return data_[i]; }

  DEV_HOST_INLINE size_t size() const { return size_; }

 private:
  T* data_;
  size_t size_;
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_WORK_SOURCE_H_
