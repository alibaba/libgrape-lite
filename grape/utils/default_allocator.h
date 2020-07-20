/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef GRAPE_UTILS_DEFAULT_ALLOCATOR_H_
#define GRAPE_UTILS_DEFAULT_ALLOCATOR_H_

#include <stdlib.h>

namespace grape {

/**
 * @brief Allocator used for grape containers, i.e., <Array>.
 *
 * @tparam _Tp
 */
template <typename _Tp>
class DefaultAllocator {
 public:
  using pointer = _Tp*;
  using size_type = size_t;
  using value_type = _Tp;

  DefaultAllocator() noexcept {}
  DefaultAllocator(const DefaultAllocator&) noexcept {}
  DefaultAllocator(DefaultAllocator&&) noexcept {}
  ~DefaultAllocator() noexcept {}

  DefaultAllocator& operator=(const DefaultAllocator&) noexcept {
    return *this;
  }
  DefaultAllocator& operator=(DefaultAllocator&&) noexcept { return *this; }

  pointer allocate(size_type __n) {
#ifdef __APPLE__
    return static_cast<pointer>(malloc(__n * sizeof(_Tp)));
#else
    return static_cast<pointer>(aligned_alloc(64, __n * sizeof(_Tp)));
#endif
  }

  void deallocate(pointer __p, size_type) { free(__p); }
};

template <typename _Tp1, typename _Tp2>
inline bool operator!=(const DefaultAllocator<_Tp1>&,
                       const DefaultAllocator<_Tp2>&) {
  return false;
}

template <typename _Tp1, typename _Tp2>
inline bool operator==(const DefaultAllocator<_Tp1>&,
                       const DefaultAllocator<_Tp2>&) {
  return true;
}

}  // namespace grape

#endif  // GRAPE_UTILS_DEFAULT_ALLOCATOR_H_
