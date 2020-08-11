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
  Vertex() {}
  explicit Vertex(T value) : value_(value) {}
  Vertex(const Vertex& rhs) : value_(rhs.value_) {}
  Vertex(Vertex&& rhs) : value_(rhs.value_) {}

  ~Vertex() {}

  inline Vertex& operator=(const Vertex& rhs) {
    value_ = rhs.value_;
    return *this;
  }

  inline Vertex& operator=(Vertex&& rhs) {
    value_ = rhs.value_;
    return *this;
  }

  inline Vertex& operator=(T value) {
    value_ = value;
    return *this;
  }

  inline Vertex& operator++() {
    value_++;
    return *this;
  }

  inline Vertex operator++(int) {
    Vertex res(value_);
    value_++;
    return res;
  }

  inline Vertex& operator--() {
    value_--;
    return *this;
  }

  inline Vertex operator--(int) {
    Vertex res(value_);
    value_--;
    return res;
  }
  inline bool operator==(const Vertex& rhs) const {
    return value_ == rhs.value_;
  }

  inline bool operator!=(const Vertex& rhs) const {
    return value_ != rhs.value_;
  }

  void Swap(Vertex& rhs) { std::swap(value_, rhs.value_); }

  inline bool operator<(const Vertex& rhs) const { return value_ < rhs.value_; }

  inline Vertex& operator*() { return *this; }

  inline T GetValue() const { return value_; }

  inline void SetValue(T value) { value_ = value; }

  friend InArchive& operator<<(InArchive& archive, const Vertex& h) {
    archive << h.value_;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive, Vertex& h) {
    archive >> h.value_;
    return archive;
  }

 private:
  T value_;
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
  VertexRange() {}
  VertexRange(T begin, T end) : begin_(begin), end_(end), size_(end - begin) {}
  VertexRange(const Vertex<T>& begin, const Vertex<T>& end)
      : begin_(begin), end_(end), size_(end.GetValue() - begin.GetValue()) {}
  VertexRange(const VertexRange& r)
      : begin_(r.begin_), end_(r.end_), size_(r.size_) {}

  inline const Vertex<T>& begin() const { return begin_; }

  inline const Vertex<T>& end() const { return end_; }

  inline size_t size() const { return size_; }

  void Swap(VertexRange& rhs) {
    begin_.Swap(rhs.begin_);
    end_.Swap(rhs.end_);
    std::swap(size_, rhs.size_);
  }

  void SetRange(T begin, T end) {
    begin_ = begin;
    end_ = end;
    size_ = end - begin;
  }

 private:
  Vertex<T> begin_, end_;
  size_t size_;
};

/**
 * @brief A discontinuous vertices collection representation. An increasing
 * labeled(but no need to be continuous) vertices must be provided to construct
 * the VertexVector.
 *
 * @tparam T Vertex ID type.
 */
template <typename T>
class VertexVector {
 public:
  VertexVector() : vertices_(dummy) {}

  explicit VertexVector(const std::vector<Vertex<T>>& vertices)
      : vertices_(vertices) {}

  inline typename std::vector<Vertex<T>>::const_iterator begin() const {
    return vertices_.get().begin();
  }

  inline typename std::vector<Vertex<T>>::const_iterator end() const {
    return vertices_.get().end();
  }

  Vertex<T> operator[](size_t idx) { return vertices_.get()[idx]; }

  Vertex<T> operator[](size_t idx) const { return vertices_.get()[idx]; }

  inline size_t size() const { return vertices_.get().size(); }

  void Swap(VertexVector& rhs) { std::swap(vertices_, rhs.vertices_); }

 private:
  std::vector<Vertex<T>> dummy;
  std::reference_wrapper<const std::vector<Vertex<T>>> vertices_;
};

template <typename T, typename VID_T>
class VertexArray : public Array<T, Allocator<T>> {
  using Base = Array<T, Allocator<T>>;

 public:
  VertexArray() : Base(), fake_start_(NULL) {}
  explicit VertexArray(const VertexRange<VID_T>& range)
      : Base(range.size()), range_(range) {
    fake_start_ = Base::data() - range_.begin().GetValue();
  }
  VertexArray(const VertexRange<VID_T>& range, const T& value)
      : Base(range.size(), value), range_(range) {
    fake_start_ = Base::data() - range_.begin().GetValue();
  }

  ~VertexArray() = default;

  void Init(const VertexRange<VID_T>& range) {
    Base::clear();
    Base::resize(range.size());
    range_ = range;
    fake_start_ = Base::data() - range_.begin().GetValue();
  }

  void Init(const VertexRange<VID_T>& range, const T& value) {
    Base::clear();
    Base::resize(range.size(), value);
    range_ = range;
    fake_start_ = Base::data() - range_.begin().GetValue();
  }

  void SetValue(VertexRange<VID_T>& range, const T& value) {
    std::fill_n(
        &Base::data()[range.begin().GetValue() - range_.begin().GetValue()],
        range.size(), value);
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
