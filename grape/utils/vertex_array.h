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

#ifndef GRAPE_UTILS_VERTEX_ARRAY_H_
#define GRAPE_UTILS_VERTEX_ARRAY_H_

#include <algorithm>
#include <utility>

#include "grape/config.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/utils/gcontainer.h"

namespace grape {

/**
 * @brief  A Vertex object only contains id of a vertex.
 * It will be used when iterating vertices of a fragment and
 * accessing data and neighbor of a vertex.
 *
 * @tparam T Vertex ID type.
 */
template <typename T>
class Vertex {
 public:
  Vertex() = default;
  DEV_HOST explicit Vertex(const T& value) noexcept : value_(value) {}

  ~Vertex() = default;

  DEV_HOST_INLINE Vertex& operator=(const T& value) noexcept {
    value_ = value;
    return *this;
  }

  DEV_HOST_INLINE Vertex& operator++() noexcept {
    value_++;
    return *this;
  }

  DEV_HOST_INLINE Vertex operator++(int) {
    Vertex res(value_);
    value_++;
    return res;
  }

  DEV_HOST_INLINE Vertex& operator--() noexcept {
    value_--;
    return *this;
  }

  DEV_HOST_INLINE Vertex operator--(int) noexcept {
    Vertex res(value_);
    value_--;
    return res;
  }

  DEV_HOST_INLINE Vertex operator+(size_t offset) const noexcept {
    Vertex res(value_ + offset);
    return res;
  }

  DEV_HOST_INLINE bool operator==(const Vertex& rhs) const {
    return value_ == rhs.value_;
  }

  DEV_HOST_INLINE bool operator!=(const Vertex& rhs) const {
    return value_ != rhs.value_;
  }

  DEV_HOST_INLINE void Swap(Vertex& rhs) {
#ifdef __CUDACC__
    thrust::swap(value_, rhs.value_);
#else
    std::swap(value_, rhs.value_);
#endif
  }

  DEV_HOST_INLINE bool operator<(const Vertex& rhs) const {
    return value_ < rhs.value_;
  }

  DEV_HOST_INLINE T GetValue() const { return value_; }

  DEV_HOST_INLINE void SetValue(T value) { value_ = value; }

  friend InArchive& operator<<(InArchive& archive, const Vertex& h) {
    archive << h.value_;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive, Vertex& h) {
    archive >> h.value_;
    return archive;
  }

 private:
  T value_{};
};

template <typename T>
bool operator<(Vertex<T> const& lhs, Vertex<T> const& rhs) {
  return lhs.GetValue() < rhs.GetValue();
}

template <typename T>
bool operator==(Vertex<T> const& lhs, Vertex<T> const& rhs) {
  return lhs.GetValue() == rhs.GetValue();
}

template <typename T>
class VertexRange {
 public:
  using vertex_t = Vertex<T>;

  DEV_HOST VertexRange() {}
  DEV_HOST VertexRange(const T& begin, const T& end)
      : begin_(begin), end_(end) {}
  DEV_HOST VertexRange(const VertexRange& r) : begin_(r.begin_), end_(r.end_) {}

  class iterator {
    using reference_type = Vertex<T>&;

   private:
    Vertex<T> cur_;

   public:
    DEV_HOST iterator() noexcept : cur_() {}
    DEV_HOST explicit iterator(const T& v) noexcept : cur_(v) {}

    DEV_HOST_INLINE reference_type operator*() noexcept { return cur_; }

    DEV_HOST_INLINE iterator& operator++() noexcept {
      ++cur_;
      return *this;
    }

    DEV_HOST_INLINE iterator operator++(int) noexcept {
      iterator ret = *this;
      ++*this;
      return ret;
    }

    DEV_HOST_INLINE iterator& operator--() noexcept {
      --cur_;
      return *this;
    }

    DEV_HOST_INLINE iterator operator--(int) noexcept {
      iterator ret = *this;
      --*this;
      return ret;
    }

    DEV_HOST_INLINE iterator operator+(size_t offset) const noexcept {
      return iterator(cur_.GetValue() + offset);
    }

    DEV_HOST bool operator==(const iterator& rhs) const noexcept {
      return cur_ == rhs.cur_;
    }

    DEV_HOST bool operator!=(const iterator& rhs) const noexcept {
      return cur_ != rhs.cur_;
    }

    DEV_HOST bool operator<(const iterator& rhs) const noexcept {
      return cur_.GetValue() < rhs.cur_.GetValue();
    }
  };

  DEV_HOST_INLINE iterator begin() const { return iterator(begin_); }

  DEV_HOST_INLINE iterator end() const { return iterator(end_); }

  DEV_HOST_INLINE size_t size() const { return end_ - begin_; }

  DEV_HOST void Swap(VertexRange& rhs) {
#ifdef __CUDACC__
    thrust::swap(begin_, rhs.begin_);
    thrust::swap(end_, rhs.end_);
#else
    std::swap(begin_, rhs.begin_);
    std::swap(end_, rhs.end_);
#endif
  }

  DEV_HOST void SetRange(const T& begin, const T& end) {
    begin_ = begin;
    end_ = end;
  }

  DEV_HOST const T& begin_value() const { return begin_; }

  DEV_HOST const T& end_value() const { return end_; }

  inline bool Contain(const Vertex<T>& v) const {
    return begin_ <= v.GetValue() && v.GetValue() < end_;
  }

  inline friend InArchive& operator<<(InArchive& in_archive,
                                      const VertexRange<T>& range) {
    in_archive << range.begin_ << range.end_;
    return in_archive;
  }

  inline friend OutArchive& operator>>(OutArchive& out_archive,
                                       VertexRange<T>& range) {
    out_archive >> range.begin_ >> range.end_;
    return out_archive;
  }

 private:
  T begin_, end_;
};

template <typename VID_T>
class DualVertexRange {
 public:
  using vertex_t = Vertex<VID_T>;

  DualVertexRange() {}

  DualVertexRange(const VID_T& head_begin, const VID_T& head_end,
                  const VID_T& tail_begin, const VID_T& tail_end) {
    SetRange(head_begin, head_end, tail_begin, tail_end);
  }

  void SetRange(const VID_T& head_begin, const VID_T& head_end,
                const VID_T& tail_begin, const VID_T& tail_end) {
    head_begin_ = head_begin;
    tail_begin_ = tail_begin;
    head_end_ = std::max(head_begin_, head_end);
    tail_end_ = std::max(tail_begin_, tail_end);

    if (head_begin_ > tail_begin_) {
      std::swap(head_begin_, tail_begin_);
      std::swap(head_end_, tail_end_);
    }
    if (head_end_ >= tail_begin_) {
      head_end_ = tail_end_;
      tail_begin_ = tail_end_;
    }
  }

  class iterator {
    using reference_type = const Vertex<VID_T>&;

   private:
    Vertex<VID_T> cur_;
    VID_T head_end_;
    VID_T tail_begin_;

   public:
    iterator() noexcept : cur_() {}
    explicit iterator(const VID_T& v) noexcept : cur_(v) {}
    explicit iterator(const VID_T& v, const VID_T& x, const VID_T& y) noexcept
        : cur_(v), head_end_(x), tail_begin_(y) {}

    inline reference_type operator*() noexcept { return cur_; }

    inline iterator& operator++() noexcept {
      ++cur_;
      if (cur_.GetValue() == head_end_) {
        cur_.SetValue(tail_begin_);
      }
      return *this;
    }

    inline iterator operator++(int) noexcept {
      VID_T new_value = cur_.GetValue() + 1;
      if (new_value == head_end_) {
        new_value = tail_begin_;
      }
      return iterator(new_value, head_end_, tail_begin_);
    }

    inline iterator& operator--() noexcept {
      if (cur_.GetValue() == tail_begin_) {
        cur_.SetValue(head_end_);
      }
      --cur_;
      return *this;
    }

    inline iterator operator--(int) noexcept {
      return iterator(cur_.GetValue()--, head_end_, tail_begin_);
    }

    iterator operator+(size_t offset) noexcept {
      VID_T new_value = cur_.GetValue() + offset;
      if (cur_.GetValue() < head_end_ && new_value >= head_end_) {
        new_value = offset - (head_end_ - cur_.GetValue()) + tail_begin_;
      }
      return iterator(new_value, head_end_, tail_begin_);
    }

    bool operator==(const iterator& rhs) noexcept { return cur_ == rhs.cur_; }

    bool operator!=(const iterator& rhs) noexcept { return cur_ != rhs.cur_; }
  };

  iterator begin() const {
    return iterator(head_begin_, head_end_, tail_begin_);
  }

  iterator end() const { return iterator(tail_end_); }

  VertexRange<VID_T> head() const {
    return VertexRange<VID_T>(head_begin_, head_end_);
  }
  VertexRange<VID_T> tail() const {
    return VertexRange<VID_T>(tail_begin_, tail_end_);
  }

  const VID_T begin_value() const { return head_begin_; }

  const VID_T end_value() const { return tail_end_; }

  bool Contain(const Vertex<VID_T>& v) const {
    return (head_begin_ <= v.GetValue() && v.GetValue() < head_end_) ||
           (tail_begin_ <= v.GetValue() && v.GetValue() < tail_end_);
  }

  VID_T size() const {
    return (head_end_ - head_begin_) + (tail_end_ - tail_begin_);
  }

  friend InArchive& operator<<(InArchive& in_archive,
                               const DualVertexRange<VID_T>& range) {
    in_archive << range.head_begin_ << range.head_end_ << range.tail_begin_
               << range.tail_end_;
    return in_archive;
  }

  friend OutArchive& operator>>(OutArchive& out_archive,
                                DualVertexRange<VID_T>& range) {
    out_archive >> range.head_begin_ >> range.head_end_ >> range.tail_begin_ >>
        range.tail_end_;
    return out_archive;
  }

 private:
  VID_T head_begin_;
  VID_T head_end_;
  VID_T tail_begin_;
  VID_T tail_end_;
};

template <typename VID_T>
inline InArchive& operator<<(InArchive& in_archive,
                             const DualVertexRange<VID_T>& range) {
  in_archive.AddBytes(&range, sizeof(DualVertexRange<VID_T>));
  return in_archive;
}

/**
 * @brief A discontinuous vertices collection representation. An increasing
 * labeled(but no need to be continuous) vertices must be provided to construct
 * the VertexVector.
 *
 * @tparam T Vertex ID type.
 */
template <typename T>
using VertexVector = std::vector<Vertex<T>>;

template <typename VERTEX_SET_T, typename T>
class VertexArray {};

template <typename VID_T, typename T>
class VertexArray<VertexRange<VID_T>, T> : public Array<T, Allocator<T>> {
  using Base = Array<T, Allocator<T>>;

 public:
  VertexArray() : Base(), fake_start_(NULL) {}
  explicit VertexArray(const VertexRange<VID_T>& range)
      : Base(range.size()), range_(range) {
    fake_start_ = Base::data() - range_.begin_value();
  }
  VertexArray(const VertexRange<VID_T>& range, const T& value)
      : Base(range.size(), value), range_(range) {
    fake_start_ = Base::data() - range_.begin_value();
  }

  ~VertexArray() = default;

  void Init(const VertexRange<VID_T>& range) {
    Base::clear();
    Base::resize(range.size());
    range_ = range;
    fake_start_ = Base::data() - range_.begin_value();
  }

  void Init(const VertexRange<VID_T>& range, const T& value) {
    Base::clear();
    Base::resize(range.size(), value);
    range_ = range;
    fake_start_ = Base::data() - range_.begin_value();
  }

  void SetValue(VertexRange<VID_T>& range, const T& value) {
    std::fill_n(&Base::data()[range.begin_value() - range_.begin_value()],
                range.size(), value);
  }
  void SetValue(const Vertex<VID_T>& loc, const T& value) {
    fake_start_[loc.GetValue()] = value;
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](const Vertex<VID_T>& loc) {
    return fake_start_[loc.GetValue()];
  }
  inline const T& operator[](const Vertex<VID_T>& loc) const {
    return fake_start_[loc.GetValue()];
  }

  void Swap(VertexArray& rhs) {
    Base::swap((Base&) rhs);
    range_.Swap(rhs.range_);
    std::swap(fake_start_, rhs.fake_start_);
  }

  void Clear() {
    VertexArray ga;
    this->Swap(ga);
  }

  const VertexRange<VID_T>& GetVertexRange() const { return range_; }

 private:
  void Resize() {}

  VertexRange<VID_T> range_;
  T* fake_start_;
};

template <typename VID_T, typename T>
class VertexArray<DualVertexRange<VID_T>, T> {
 public:
  VertexArray() : head_(), tail_() {}
  explicit VertexArray(const DualVertexRange<VID_T>& range)
      : head_(range.head()), tail_(range.tail()) {
    initMid();
  }
  VertexArray(const DualVertexRange<VID_T>& range, const T& value)
      : head_(range.head(), value), tail_(range.tail(), value) {
    initMid();
  }
  ~VertexArray() = default;

  void Init(const VertexRange<VID_T>& range) {
    head_.Init(range);
    tail_.Init(VertexRange<VID_T>(mid_, mid_));
    initMid();
  }

  void Init(const DualVertexRange<VID_T>& range) {
    head_.Init(range.head());
    tail_.Init(range.tail());
    initMid();
  }

  void Init(const VertexRange<VID_T>& range, const T& value) {
    head_.Init(range, value);
    tail_.Init(VertexRange<VID_T>(mid_, mid_));
    initMid();
  }

  void Init(const DualVertexRange<VID_T>& range, const T& value) {
    head_.Init(range.head(), value);
    tail_.Init(range.tail(), value);
    initMid();
  }

  inline T& operator[](const Vertex<VID_T>& loc) {
    return loc.GetValue() < mid_ ? head_[loc] : tail_[loc];
  }

  inline const T& operator[](const Vertex<VID_T>& loc) const {
    return loc.GetValue() < mid_ ? head_[loc] : tail_[loc];
  }

  void Swap(VertexArray& rhs) {
    head_.Swap(rhs.head_);
    tail_.Swap(rhs.tail_);
    std::swap(mid_, rhs.mid_);
  }

  void Clear() {
    head_.Clear();
    tail_.Clear();
  }

  void SetValue(const T& value) {
    head_.SetValue(value);
    tail_.SetValue(value);
  }

 private:
  void initMid() { mid_ = head_.GetVertexRange().end_value(); }

  VertexArray<VertexRange<VID_T>, T> head_;
  VertexArray<VertexRange<VID_T>, T> tail_;
  VID_T mid_;
};

}  // namespace grape

namespace std {
template <typename T>
struct hash<grape::Vertex<T>> {
  inline size_t operator()(const grape::Vertex<T>& obj) const {
    return hash<T>()(obj.GetValue());
  }
};

}  // namespace std

#endif  // GRAPE_UTILS_VERTEX_ARRAY_H_
