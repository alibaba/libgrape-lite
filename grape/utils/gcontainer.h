/**
 *
 * NOLINT(legal/copyright)
 *
 * The file utils/gcontainer.h is based on code from libcxx,
 *
 *    https://github.com/llvm-mirror/libcxx/blob/master/include/vector
 *
 * which has the following license:
 *
// -*- C++ -*-
//===------------------------------ vector --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
*/

#ifndef GRAPE_UTILS_GCONTAINER_H_
#define GRAPE_UTILS_GCONTAINER_H_

#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include "grape/config.h"
#include "grape/types.h"

namespace grape {

namespace internal {
template <class _Alloc, class _Iter, class _Ptr>
inline void __construct_range_forward(_Alloc& __a, _Iter __begin1, _Iter __end1,
                                      _Ptr& __begin2) {
  for (; __begin1 != __end1; ++__begin1, (void) ++__begin2) {
    std::allocator_traits<_Alloc>::construct(__a, __begin2, *__begin1);
  }
}

template <class _Alloc, class _SourceTp, class _DestTp,
          class _RawSourceTp = typename std::remove_const<_SourceTp>::type,
          class _RawDestTp = typename std::remove_const<_DestTp>::type>
inline typename std::enable_if<
#if __GNUG__ && __GNUC__ < 5
    // is_trivially_move_constructible is part of C++11 but not available in
    // gcc 4.8/4.9
    std::is_pod<_DestTp>::value
#else
    std::is_trivially_move_constructible<_DestTp>::value
#endif
        && std::is_same<_RawSourceTp, _RawDestTp>::value,
    void>::type
__construct_range_forward(_Alloc&, _SourceTp* __begin1, _SourceTp* __end1,
                          _DestTp*& __begin2) {
  std::ptrdiff_t _Np = __end1 - __begin1;
  if (_Np > 0) {
    std::memcpy(const_cast<_RawDestTp*>(__begin2), __begin1,
                _Np * sizeof(_DestTp));
    __begin2 += _Np;
  }
}
}  // namespace internal

/**
 * @brief Array a std::vector-like container type without reserving memory.
 *
 * Unlike std::array, Array is resizable, and unlike std::vector, Array takes
 * exactly memory for elements without reserving spaces for further insertions.
 *
 * @tparam _Tp Type of elements in the array.
 * @tparam _Alloc Allocator type that will be used for memory allocation.
 */
template <typename _Tp, typename _Alloc = Allocator<_Tp>>
class Array {
 public:
  using pointer = _Tp*;
  using const_pointer = const _Tp*;
  using reference = _Tp&;
  using const_reference = const _Tp&;
  using size_type = size_t;
  using value_type = _Tp;
  using allocator_type = _Alloc;

  using iterator = pointer;
  using const_iterator = const_pointer;

  using __alloc_traits = std::allocator_traits<_Alloc>;

  struct __Array_base : public _Alloc {
    pointer __begin_;
    pointer __end_;

    __Array_base() noexcept : _Alloc(), __begin_(nullptr), __end_(nullptr) {}
    explicit __Array_base(const _Alloc& __a) noexcept
        : _Alloc(__a), __begin_(nullptr), __end_(nullptr) {}
    explicit __Array_base(_Alloc&& __a) noexcept
        : _Alloc(std::move(__a)), __begin_(nullptr), __end_(nullptr) {}

    size_type size() const noexcept { return size_type(__end_ - __begin_); }
    bool empty() const noexcept { return __end_ == __begin_; }
  };

 private:
  struct _ConstructTransaction {
    explicit _ConstructTransaction(__Array_base& __v, size_type __n)
        : __v_(__v), __pos_(__v.__end_), __new_end_(__v.__end_ + __n) {}
    ~_ConstructTransaction() { __v_.__end_ = __pos_; }

    __Array_base& __v_;
    pointer __pos_;
    const_pointer const __new_end_;

   private:
    _ConstructTransaction(_ConstructTransaction const&) = delete;
    _ConstructTransaction& operator=(_ConstructTransaction const&) = delete;
  };

  __Array_base __base;

  allocator_type& __alloc() noexcept {
    return *static_cast<allocator_type*>(&this->__base);
  }

  const allocator_type& __alloc() const noexcept {
    return *static_cast<const allocator_type*>(&this->__base);
  }

  inline void __vallocate(size_type __n) {
    this->__base.__begin_ = this->__base.__end_ =
        __alloc_traits::allocate(this->__alloc(), __n);
  }

  inline void __vdeallocate() {
    if (this->__base.__begin_ != nullptr) {
      __alloc_traits::deallocate(this->__alloc(), this->__base.__begin_,
                                 size());
      this->__base.__begin_ = this->__base.__end_ = nullptr;
    }
  }

  inline void __vdeallocate(pointer __begin, size_type __size) {
    if (__begin != nullptr) {
      __alloc_traits::deallocate(this->__alloc(), __begin, __size);
    }
  }

  inline void __construct_at_end(size_type __n) {
    _ConstructTransaction __tx(this->__base, __n);
    for (; __tx.__pos_ != __tx.__new_end_; ++__tx.__pos_) {
      __alloc_traits::construct(this->__alloc(), __tx.__pos_);
    }
  }

  inline void __construct_at_end(size_type __n, const_reference __x) {
    _ConstructTransaction __tx(this->__base, __n);
    for (; __tx.__pos_ != __tx.__new_end_; ++__tx.__pos_) {
      __alloc_traits::construct(this->__alloc(), __tx.__pos_, __x);
    }
  }

  template <class _ForwardIterator>
  inline void __construct_at_end(_ForwardIterator __first,
                                 _ForwardIterator __last, size_type __n) {
    _ConstructTransaction __tx(this->__base, __n);
    internal::__construct_range_forward(this->__alloc(), __first, __last,
                                        __tx.__pos_);
  }

  inline void __destruct_at_end(pointer __new_last) {
    pointer __soon_to_be_end = this->__base.__end_;
    while (__new_last != __soon_to_be_end) {
      __alloc_traits::destroy(__alloc(), --__soon_to_be_end);
    }
  }

  inline void __destruct_at_end(pointer __new_last, pointer __end) {
    pointer __soon_to_be_end = __end;
    while (__new_last != __soon_to_be_end) {
      __alloc_traits::destroy(__alloc(), --__soon_to_be_end);
    }
  }

  void __range_check(size_type __n) const noexcept {
    CHECK(__n < this->size());
  }

 public:
  allocator_type get_allocator() const noexcept {
    return allocator_type(__alloc());
  }

  Array() noexcept : __base() {}
  explicit Array(const allocator_type& __a) noexcept : __base(__a) {}
  explicit Array(size_type __n, const allocator_type& __a = allocator_type())
      : __base(__a) {
    if (__n > 0) {
      __vallocate(__n);
      __construct_at_end(__n);
    }
  }
  Array(size_type __n, const value_type& __x,
        const allocator_type& __a = allocator_type())
      : __base(__a) {
    if (__n > 0) {
      __vallocate(__n);
      __construct_at_end(__n, __x);
    }
  }

  Array(const Array& __x)
      : __base(__alloc_traits::select_on_container_copy_construction(
            __x.__alloc())) {
    size_type __n = __x.size();
    if (__n > 0) {
      __vallocate(__n);
      __construct_at_end(__x.__base.__begin_, __x.__base.__end_, __n);
    }
  }

  Array(Array&& __x) noexcept : __base(std::move(__x.__alloc())) {
    this->__base.__begin_ = __x.__base.__begin_;
    this->__base.__end_ = __x.__base.__end_;
    __x.__base.__begin_ = __x.__base.__end_ = nullptr;
  }

  Array(const Array& __x, const allocator_type& __a) : __base(__a) {
    size_type __n = __x.size();
    if (__n > 0) {
      __vallocate(__n);
      __construct_at_end(__x.__base.__begin_, __x.__base.__end_, __n);
    }
  }

  Array(Array&& __x, const allocator_type& __a) : __base(__a) {
    if (__a == __x.__alloc()) {
      this->__base.__begin_ = __x.__base.__begin_;
      this->__base.__end_ = __x.__base.__end_;
      __x.__base.__begin_ = __x.__base.__end_ = nullptr;
    } else {
      size_type __n = __x.size();
      __vallocate(__n);
      __construct_at_end(__x.__base.__begin_, __x.__base.__end_, __n);
      __x.clear();
    }
  }

  ~Array() noexcept { clear(); }

  Array& operator=(const Array& __x) {
    if (&__x == this) {
      return *this;
    }
    __destruct_at_end(this->__base.__begin_);

    if (__x.__alloc() != this->__alloc()) {
      __vdeallocate();
      this->__alloc() = __x.__alloc();
    }
    size_type __n = __x.size();
    if (size() != __n) {
      __vdeallocate();
      __vallocate(__n);
    }
    __construct_at_end(__x.__base.__begin_, __x.__base.__end_, __n);
    return *this;
  }

  Array& operator=(Array&& __x) {
    __destruct_at_end(this->__base.__begin_);
    __vdeallocate();
    if (__x.__alloc() == this->__alloc()) {
      this->__base.__begin_ = __x.__base.__begin_;
      this->__base.__end_ = __x.__base.__end_;
      __x.__base.__begin_ = __x.__base.__end_ = nullptr;
    } else {
      this->__alloc() = __x.__alloc();
      size_type __n = __x.size();
      __vallocate(__n);
      __construct_at_end(this->__base.__begin_, this->__base.__end_, __n);
      __x.clear();
    }
    return *this;
  }

  size_type size() const noexcept { return this->__base.size(); }

  void resize(size_type __new_size) {
    const size_type __old_size = size();
    if (__old_size > __new_size) {
      pointer __old_begin = this->__base.__begin_;
      pointer __old_end = this->__base.__end_;
      __vallocate(__new_size);
      __construct_at_end(__old_begin, __old_begin + __new_size, __new_size);

      __destruct_at_end(__old_begin, __old_end);
      __vdeallocate(__old_begin, __old_size);
    } else if (__old_size < __new_size) {
      pointer __old_begin = this->__base.__begin_;
      pointer __old_end = this->__base.__end_;
      __vallocate(__new_size);
      __construct_at_end(__old_begin, __old_end, __old_size);
      __construct_at_end(__new_size - __old_size);

      __destruct_at_end(__old_begin, __old_end);
      __vdeallocate(__old_begin, __old_size);
    }
  }

  void resize(size_type __new_size, const value_type& __x) {
    const size_type __old_size = size();
    if (__old_size > __new_size) {
      pointer __old_begin = this->__base.__begin_;
      pointer __old_end = this->__base.__end_;
      __vallocate(__new_size);
      __construct_at_end(__old_begin, __old_begin + __new_size, __new_size);

      __destruct_at_end(__old_begin, __old_end);
      __vdeallocate(__old_begin, __old_size);
    } else if (__old_size < __new_size) {
      pointer __old_begin = this->__base.__begin_;
      pointer __old_end = this->__base.__end_;
      __vallocate(__new_size);
      __construct_at_end(__old_begin, __old_end, __old_size);
      __construct_at_end(__new_size - __old_size, __x);

      __destruct_at_end(__old_begin, __old_end);
      __vdeallocate(__old_begin, __old_size);
    }
  }

  bool empty() const noexcept { return this->__base.empty(); }

  reference operator[](size_type __n) noexcept {
    return *(this->__base.__begin_ + __n);
  }

  const_reference operator[](size_type __n) const noexcept {
    return *(this->__base.__begin_ + __n);
  }

  reference at(size_type __n) noexcept {
    __range_check(__n);
    return *(this->__base.__begin_ + __n);
  }

  const_reference at(size_type __n) const noexcept {
    __range_check(__n);
    return *(this->__base.__begin_ + __n);
  }

  pointer data() noexcept { return this->__base.__begin_; }
  const_pointer data() const noexcept { return this->__base.__begin_; }

  iterator begin() noexcept { return iterator(this->__base.__begin_); }
  const_iterator begin() const noexcept {
    return const_iterator(this->__base.__begin_);
  }

  iterator end() noexcept { return iterator(this->__base.__end_); }
  const_iterator end() const noexcept {
    return const_iterator(this->__base.__end_);
  }

  void swap(Array& __x) noexcept {
    std::swap(this->__alloc(), __x.__alloc());
    std::swap(this->__base.__begin_, __x.__base.__begin_);
    std::swap(this->__base.__end_, __x.__base.__end_);
  }

  void clear() noexcept {
    __destruct_at_end(this->__base.__begin_);
    __vdeallocate();
  }
};

/**
 * @brief Template specialization of Array for EmptyType, without consuming
 * extra memory for EmptyType but provides same interfaces with Array of usual
 * data types.
 *
 * @tparam _Alloc Allocator type that will be used for memory allocation.
 */
template <typename _Alloc>
class Array<EmptyType, _Alloc> {
 public:
  using pointer = EmptyType*;
  using const_pointer = const EmptyType*;
  using reference = EmptyType&;
  using const_reference = const EmptyType&;
  using size_type = size_t;
  using value_type = EmptyType;
  using allocator_type = _Alloc;

  struct __Array_base : public _Alloc {
    __Array_base() noexcept : _Alloc() {}
    explicit __Array_base(const _Alloc& __a) noexcept : _Alloc(__a) {}
    explicit __Array_base(_Alloc&& __a) noexcept : _Alloc(std::move(__a)) {}
  };

 private:
  __Array_base __base;
  EmptyType __val;
  size_type __size;

  allocator_type& __alloc() noexcept {
    return *static_cast<allocator_type*>(&this->__base);
  }

  const allocator_type& __alloc() const noexcept {
    return *static_cast<const allocator_type*>(&this->__base);
  }

  void __range_check(size_type __n) const noexcept {
    CHECK(__n < this->size());
  }

 public:
  allocator_type get_allocator() const noexcept {
    return allocator_type(__alloc());
  }

  Array() noexcept : __base(), __size(0) {}
  explicit Array(const allocator_type& __a) noexcept : __base(__a), __size(0) {}
  explicit Array(size_type __n, const allocator_type& __a = allocator_type())
      : __base(__a), __size(__n) {}
  Array(size_type __n, const value_type& __value,
        const allocator_type& __a = allocator_type())
      : __base(__a), __size(__n) {}

  Array(const Array& __x) : __base(__x.__alloc()), __size(__x.__size) {}

  Array(Array&& __x) noexcept
      : __base(std::move(__x.__alloc())), __size(__x.__size) {
    __x.__size = 0;
  }

  Array(const Array& __x, const allocator_type& __a)
      : __base(__x.__base), __size(__x.__size) {}

  Array(Array&& __x, const allocator_type& __m)
      : __base(__m), __size(__x.__size) {
    __x.__size = 0;
  }

  ~Array() noexcept {}

  Array& operator=(const Array& __x) {
    if (&__x == this) {
      return *this;
    }
    this->__alloc() = __x.__alloc();
    __size = __x.__size;
    return *this;
  }

  Array& operator=(Array&& __x) {
    this->__alloc() = std::move(__x.__alloc());
    __size = __x.__size;
    __x.__size = 0;
    return *this;
  }

  size_type size() const noexcept { return this->__size; }

  void resize(size_type __new_size) { this->__size = __new_size; }

  void resize(size_type __new_size, const value_type&) {
    this->__size = __new_size;
  }

  bool empty() const noexcept { return this->__size == 0; }

  reference operator[](size_type) noexcept { return this->__val; }

  const_reference operator[](size_type) const noexcept { return this->__val; }

  reference at(size_type __n) noexcept {
    __range_check(__n);
    return this->__val;
  }

  const_reference at(size_type __n) const noexcept {
    __range_check(__n);
    return this->__val;
  }

  pointer data() noexcept { return nullptr; }
  const_pointer data() const noexcept { return nullptr; }

  struct iterator {
    iterator() noexcept = default;
    explicit iterator(EmptyType* val, size_t index) noexcept
        : val_(val), index_(index) {}

    reference operator*() noexcept { return *val_; }
    pointer operator->() noexcept { return val_; }

    iterator& operator++() noexcept {
      ++index_;
      return *this;
    }
    iterator operator++(int) noexcept { return iterator(index_++); }

    iterator& operator--() noexcept {
      --index_;
      return *this;
    }
    iterator operator--(int) noexcept { return iterator(index_--); }

    friend bool operator==(const iterator& lhs, const iterator& rhs) {
      return lhs.index_ == rhs.index_;
    }

    friend bool operator!=(const iterator& lhs, const iterator& rhs) {
      return lhs.index_ != rhs.index_;
    }

   private:
    EmptyType* val_;
    size_t index_;
  };

  struct const_iterator {
    const_iterator() noexcept = default;
    explicit const_iterator(EmptyType* val, size_t index) noexcept
        : val_(val), index_(index) {}

    const_reference operator*() const noexcept { return *val_; }
    const_pointer operator->() const noexcept { return val_; }

    const_iterator& operator++() noexcept {
      ++index_;
      return *this;
    }
    const_iterator operator++(int) noexcept { return const_iterator(index_++); }

    const_iterator& operator--() noexcept {
      --index_;
      return *this;
    }
    const_iterator operator--(int) noexcept { return const_iterator(index_--); }

    friend bool operator==(const const_iterator& lhs,
                           const const_iterator& rhs) {
      return lhs.index_ == rhs.index_;
    }

    friend bool operator!=(const const_iterator& lhs,
                           const const_iterator& rhs) {
      return lhs.index_ != rhs.index_;
    }

   private:
    EmptyType* val_;
    size_t index_;
  };

  iterator begin() noexcept { return iterator(&__val, 0); }
  const_iterator begin() const noexcept { return const_iterator(&__val, 0); }

  iterator end() noexcept { return iterator(&__val, this->__size); }
  const_iterator end() const noexcept {
    return const_iterator(&__val, this->__size);
  }

  void swap(Array& __x) noexcept {
    std::swap(this->__alloc(), __x.__alloc());
    std::swap(this->__size, __x.__size);
  }

  void clear() noexcept { this->__size = 0; }
};

}  // namespace grape

#endif  // GRAPE_UTILS_GCONTAINER_H_
