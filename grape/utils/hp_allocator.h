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

#ifndef GRAPE_UTILS_HP_ALLOCATOR_H_
#define GRAPE_UTILS_HP_ALLOCATOR_H_

#include <stdlib.h>
#include <sys/mman.h>

#include <glog/logging.h>

#ifdef USE_HUGEPAGES

#ifdef __ia64__
#define ADDR (void*) (0x8000000000000000UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_FIXED)
#else
#define ADDR (void*) (0x0UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#endif

#define PROTECTION (PROT_READ | PROT_WRITE)

#define HUGEPAGE_SIZE (2UL * 1024 * 1024)
#define HUGEPAGE_MASK (2UL * 1024 * 1024 - 1UL)
#define ROUND_UP(size) (((size) + HUGEPAGE_MASK) & (~HUGEPAGE_MASK))

namespace grape {

/**
 * @brief Allocator used for grape containers, i.e., <Array>.
 *
 * @tparam _Tp
 */
template <typename _Tp>
class HpAllocator {
 public:
  typedef _Tp value_type;
  typedef _Tp* pointer;
  typedef const _Tp* const_pointer;
  typedef _Tp& reference;
  typedef const _Tp& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  template <typename _Tp2>
  struct rebind {
    typedef HpAllocator<_Tp2> other;
  };

  HpAllocator() noexcept {}
  HpAllocator(const HpAllocator&) noexcept {}
  HpAllocator(HpAllocator&&) noexcept {}
  ~HpAllocator() noexcept {}

  HpAllocator& operator=(const HpAllocator&) noexcept { return *this; }
  HpAllocator& operator=(HpAllocator&&) noexcept { return *this; }

  pointer address(reference value) const {
    return &value;
  }

  const_pointer address(const_reference value) const {
    return &value;
  }

  size_type max_size() const {
    return std::numeric_limits<std::size_t>::max() / sizeof(_Tp);
  }

  pointer allocate(size_type __n) {
    void* addr =
        mmap(ADDR, ROUND_UP(__n * sizeof(_Tp)), PROTECTION, FLAGS, -1, 0);
    if (addr == MAP_FAILED) {
      VLOG(1) << "Allocating hugepages failed, using malloc";
#ifdef __APPLE__
      addr = malloc(__n * sizeof(_Tp));
#else
      addr = aligned_alloc(64, __n * sizeof(_Tp));
#endif
    }
    return static_cast<pointer>(addr);
  }

  void construct(pointer p, const _Tp& value) {
    new((void*)p)_Tp(value);
  }

  void destroy(pointer p) {
    p->~_Tp();
  }

  void deallocate(pointer __p, size_type __n) {
    if (munmap(__p, ROUND_UP(__n * sizeof(_Tp)))) {
      perror("deallocator");
      free(__p);
    }
  }
};

template <typename _Tp1, typename _Tp2>
inline bool operator!=(const HpAllocator<_Tp1>&, const HpAllocator<_Tp2>&) {
  return false;
}

template <typename _Tp1, typename _Tp2>
inline bool operator==(const HpAllocator<_Tp1>&, const HpAllocator<_Tp2>&) {
  return true;
}

}  // namespace grape

#undef ADDR
#undef FLAGS
#undef HUGEPAGE_SIZE
#undef HUGEPAGE_MASK
#undef ROUND_UP

#endif

#endif  // GRAPE_UTILS_HP_ALLOCATOR_H_
