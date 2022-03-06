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

#ifndef GRAPE_UTILS_VERTEX_SET_H_
#define GRAPE_UTILS_VERTEX_SET_H_

#include <utility>

#include "grape/utils/bitset.h"
#include "grape/utils/thread_pool.h"
#include "grape/utils/vertex_array.h"

namespace grape {
/**
 * @brief A vertex set with dense vertices.
 *
 * @tparam VERTEX_SET_T Vertex set type.
 */
template <typename VERTEX_SET_T>
class DenseVertexSet {};

template <typename VID_T>
class DenseVertexSet<VertexRange<VID_T>> {
 public:
  DenseVertexSet() = default;

  explicit DenseVertexSet(const VertexRange<VID_T>& range)
      : beg_(range.begin_value()), end_(range.end_value()), bs_(end_ - beg_) {}

  ~DenseVertexSet() = default;

  void Init(const VertexRange<VID_T>& range, ThreadPool& thread_pool) {
    beg_ = range.begin_value();
    end_ = range.end_value();
    bs_.init(end_ - beg_);
    bs_.parallel_clear(thread_pool);
  }

  void Init(const VertexVector<VID_T>& vertices, ThreadPool& thread_pool) {
    if (vertices.size() == 0)
      return;
    beg_ = vertices[0].GetValue();
    end_ = vertices[vertices.size() - 1].GetValue();
    bs_.init(end_ - beg_ + 1);
    bs_.parallel_clear(thread_pool);
  }

  void Init(const VertexRange<VID_T>& range) {
    beg_ = range.begin_value();
    end_ = range.end_value();
    bs_.init(end_ - beg_);
    bs_.clear();
  }

  void Init(const VertexVector<VID_T>& vertices) {
    if (vertices.size() == 0)
      return;
    beg_ = vertices[0].GetValue();
    end_ = vertices[vertices.size() - 1].GetValue();
    bs_.init(end_ - beg_ + 1);
    bs_.clear();
  }

  void Insert(Vertex<VID_T> u) { bs_.set_bit(u.GetValue() - beg_); }

  bool InsertWithRet(Vertex<VID_T> u) {
    return bs_.set_bit_with_ret(u.GetValue() - beg_);
  }

  void Erase(Vertex<VID_T> u) { bs_.reset_bit(u.GetValue() - beg_); }

  bool EraseWithRet(Vertex<VID_T> u) {
    return bs_.reset_bit_with_ret(u.GetValue() - beg_);
  }

  bool Exist(Vertex<VID_T> u) const { return bs_.get_bit(u.GetValue() - beg_); }

  VertexRange<VID_T> Range() const { return VertexRange<VID_T>(beg_, end_); }

  size_t Count() const { return bs_.count(); }

  size_t ParallelCount(ThreadPool& thread_pool) const {
    return bs_.parallel_count(thread_pool);
  }

  size_t PartialCount(VID_T beg, VID_T end) const {
    return bs_.partial_count(beg - beg_, end - beg_);
  }

  size_t ParallelPartialCount(ThreadPool& thread_pool, VID_T beg,
                              VID_T end) const {
    return bs_.parallel_partial_count(thread_pool, beg - beg_, end - beg_);
  }

  void Clear() { bs_.clear(); }

  void ParallelClear(ThreadPool& thread_pool) {
    bs_.parallel_clear(thread_pool);
  }

  void Swap(DenseVertexSet& rhs) {
    std::swap(beg_, rhs.beg_);
    std::swap(end_, rhs.end_);
    bs_.swap(rhs.bs_);
  }

  Bitset& GetBitset() { return bs_; }

  const Bitset& GetBitset() const { return bs_; }

  bool Empty() const { return bs_.empty(); }

  bool PartialEmpty(VID_T beg, VID_T end) const {
    return bs_.partial_empty(beg - beg_, end - beg_);
  }

 private:
  VID_T beg_;
  VID_T end_;
  Bitset bs_;
};

template <typename VID_T>
class DenseVertexSet<DualVertexRange<VID_T>> {
 public:
  DenseVertexSet() = default;

  explicit DenseVertexSet(const DualVertexRange<VID_T>& range)
      : head_beg_(range.head().begin_value()),
        head_end_(range.head().end_value()),
        tail_beg_(range.tail().begin_value()),
        tail_end_(range.tail().end_value()),
        head_bs_(head_end_ - head_beg_),
        tail_bs_(tail_end_ - tail_beg_) {}

  ~DenseVertexSet() = default;

  void Init(const VertexRange<VID_T>& range, ThreadPool& thread_pool) {
    head_beg_ = range.head().begin_value();
    head_end_ = range.head().end_value();
    tail_beg_ = range.tail().begin_value();
    tail_end_ = range.tail().end_value();
    head_bs_.init(head_end_ - head_beg_);
    tail_bs_.init(tail_end_ - tail_beg_);
    head_bs_.parallel_clear(thread_pool);
    tail_bs_.parallel_clear(thread_pool);
  }

  void Init(const VertexVector<VID_T>& vertices, ThreadPool& thread_pool) {
    if (vertices.size() == 0)
      return;
    head_beg_ = vertices[0].GetValue();
    head_end_ = vertices[vertices.size() - 1].GetValue();
    head_bs_.init(head_end_ - head_beg_ + 1);
    head_bs_.parallel_clear(thread_pool);
    tail_beg_ = head_end_;
    tail_end_ = tail_beg_;
  }

  void Init(const DualVertexRange<VID_T>& range) {
    head_beg_ = range.head().begin_value();
    head_end_ = range.head().end_value();
    tail_beg_ = range.tail().begin_value();
    tail_end_ = range.tail().end_value();
    head_bs_.init(head_end_ - head_beg_);
    tail_bs_.init(tail_end_ - tail_beg_);
    head_bs_.clear();
    tail_bs_.clear();
  }

  void Init(const VertexVector<VID_T>& vertices) {
    if (vertices.size() == 0)
      return;
    head_beg_ = vertices[0].GetValue();
    head_end_ = vertices[vertices.size() - 1].GetValue();
    head_bs_.init(head_end_ - head_beg_ + 1);
    head_bs_.clear();
    tail_beg_ = head_end_;
    tail_end_ = tail_beg_;
  }

  void Insert(Vertex<VID_T> u) {
    if (u.GetValue() < head_end_) {
      head_bs_.set_bit(u.GetValue() - head_beg_);
    } else {
      tail_bs_.set_bit(u.GetValue() - tail_beg_);
    }
  }

  bool InsertWithRet(Vertex<VID_T> u) {
    if (u.GetValue() < head_end_) {
      return head_bs_.set_bit_with_ret(u.GetValue() - head_beg_);
    } else {
      return tail_bs_.set_bit_with_ret(u.GetValue() - tail_beg_);
    }
  }

  void Erase(Vertex<VID_T> u) {
    if (u.GetValue() < head_end_) {
      head_bs_.reset_bit(u.GetValue() - head_beg_);
    } else {
      tail_bs_.reset_bit(u.GetValue() - tail_beg_);
    }
  }

  bool EraseWithRet(Vertex<VID_T> u) {
    if (u.GetValue() < head_end_) {
      return head_bs_.reset_bit_with_ret(u.GetValue() - head_beg_);
    } else {
      return tail_bs_.reset_bit_with_ret(u.GetValue() - tail_beg_);
    }
  }

  bool Exist(Vertex<VID_T> u) const {
    if (u.GetValue() < head_end_) {
      return head_bs_.get_bit(u.GetValue() - head_beg_);
    } else {
      return tail_bs_.get_bit(u.GetValue() - tail_beg_);
    }
  }

  DualVertexRange<VID_T> Range() const {
    return DualVertexRange<VID_T>(head_beg_, head_end_, tail_beg_, tail_end_);
  }

  size_t Count() const { return head_bs_.count() + tail_bs_.count(); }

  size_t ParallelCount(ThreadPool& thread_pool) const {
    return head_bs_.parallel_count(thread_pool) +
           tail_bs_.parallel_count(thread_pool);
  }

  size_t PartialCount(VID_T beg, VID_T end) const {
    size_t ret = 0;
    if (beg < head_end_) {
      ret += head_bs_.partial_count(std::max(beg, head_beg_) - head_beg_,
                                    std::min(end, head_end_) - head_beg_);
    }
    if (end > tail_beg_) {
      ret += tail_bs_.partial_count(std::max(beg, tail_beg_) - tail_beg_,
                                    std::min(end, tail_end_) - tail_beg_);
    }
    return ret;
  }

  size_t ParallelPartialCount(ThreadPool& thread_pool, VID_T beg,
                              VID_T end) const {
    size_t ret = 0;
    if (beg < head_end_) {
      ret += head_bs_.parallel_partial_count(
          thread_pool, std::max(beg, head_beg_) - head_beg_,
          std::min(end, head_end_) - head_beg_);
    }
    if (end > tail_beg_) {
      ret += tail_bs_.parallel_partial_count(
          thread_pool, std::max(beg, tail_beg_) - tail_beg_,
          std::min(end, tail_end_) - tail_beg_);
    }
    return ret;
  }

  void Clear() {
    head_bs_.clear();
    tail_bs_.clear();
  }

  void ParallelClear(ThreadPool& thread_pool) {
    head_bs_.parallel_clear(thread_pool);
    tail_bs_.parallel_clear(thread_pool);
  }

  void Swap(DenseVertexSet& rhs) {
    std::swap(head_beg_, rhs.head_beg_);
    std::swap(head_end_, rhs.head_end_);
    std::swap(tail_beg_, rhs.tail_beg_);
    std::swap(tail_end_, rhs.tail_end_);
    head_bs_.swap(rhs.head_bs_);
    tail_bs_.swap(rhs.tail_bs_);
  }

  Bitset& GetHeadBitset() { return head_bs_; }

  const Bitset& GetHeadBitset() const { return head_bs_; }

  Bitset& GetTailBitset() { return tail_bs_; }

  const Bitset& GetTailBitset() const { return tail_bs_; }

  bool Empty() const { return head_bs_.empty() && tail_bs_.empty(); }

  bool PartialEmpty(VID_T beg, VID_T end) const {
    if (beg < head_end_) {
      if (!head_bs_.partial_empty(std::max(beg, head_beg_) - head_beg_,
                                  std::min(end, head_end_) - head_beg_)) {
        return false;
      }
    }
    if (end > tail_beg_) {
      if (!tail_bs_.partial_empty(std::max(beg, tail_beg_) - tail_beg_,
                                  std::min(end, tail_end_) - tail_beg_)) {
        return false;
      }
    }
    return true;
  }

 private:
  VID_T head_beg_;
  VID_T head_end_;
  VID_T tail_beg_;
  VID_T tail_end_;
  Bitset head_bs_;
  Bitset tail_bs_;
};

template <typename VID_T>
class DenseVertexSet<VertexVector<VID_T>> {
 public:
  DenseVertexSet() = default;

  explicit DenseVertexSet(const VertexRange<VID_T>& range)
      : beg_(range.begin_value()), end_(range.end_value()), bs_(end_ - beg_) {}

  explicit DenseVertexSet(const VertexVector<VID_T>& vertices) {
    if (vertices.size() == 0) {
      beg_ = 0;
      end_ = 0;
    } else {
      beg_ = vertices[0].GetValue();
      end_ = vertices[vertices.size() - 1].GetValue();
      bs_.init(end_ - beg_ + 1);
      bs_.clear();
    }
  }

  ~DenseVertexSet() = default;

  void Init(const VertexRange<VID_T>& range, ThreadPool& thread_pool) {
    beg_ = range.begin_value();
    end_ = range.end_value();
    bs_.init(end_ - beg_);
    bs_.parallel_clear(thread_pool);
  }

  void Init(const VertexVector<VID_T>& vertices, ThreadPool& thread_pool) {
    if (vertices.size() == 0)
      return;
    beg_ = vertices[0].GetValue();
    end_ = vertices[vertices.size() - 1].GetValue();
    bs_.init(end_ - beg_ + 1);
    bs_.parallel_clear(thread_pool);
  }

  void Init(const VertexRange<VID_T>& range) {
    beg_ = range.begin_value();
    end_ = range.end_value();
    bs_.init(end_ - beg_);
    bs_.clear();
  }

  void Init(const VertexVector<VID_T>& vertices) {
    if (vertices.size() == 0)
      return;
    beg_ = vertices[0].GetValue();
    end_ = vertices[vertices.size() - 1].GetValue();
    bs_.init(end_ - beg_ + 1);
    bs_.clear();
  }

  void Insert(Vertex<VID_T> u) { bs_.set_bit(u.GetValue() - beg_); }

  bool InsertWithRet(Vertex<VID_T> u) {
    return bs_.set_bit_with_ret(u.GetValue() - beg_);
  }

  void Erase(Vertex<VID_T> u) { bs_.reset_bit(u.GetValue() - beg_); }

  bool EraseWithRet(Vertex<VID_T> u) {
    return bs_.reset_bit_with_ret(u.GetValue() - beg_);
  }

  bool Exist(Vertex<VID_T> u) const { return bs_.get_bit(u.GetValue() - beg_); }

  VertexRange<VID_T> Range() const { return VertexRange<VID_T>(beg_, end_); }

  size_t Count() const { return bs_.count(); }

  size_t ParallelCount(ThreadPool& thread_pool) const {
    return bs_.parallel_count(thread_pool);
  }

  size_t PartialCount(VID_T beg, VID_T end) const {
    return bs_.partial_count(beg - beg_, end - beg_);
  }

  size_t ParallelPartialCount(ThreadPool& thread_pool, VID_T beg,
                              VID_T end) const {
    return bs_.parallel_partial_count(thread_pool, beg - beg_, end - beg_);
  }

  void Clear() { bs_.clear(); }

  void ParallelClear(ThreadPool& thread_pool) {
    bs_.parallel_clear(thread_pool);
  }

  void Swap(DenseVertexSet& rhs) {
    std::swap(beg_, rhs.beg_);
    std::swap(end_, rhs.end_);
    bs_.swap(rhs.bs_);
  }

  Bitset& GetBitset() { return bs_; }

  const Bitset& GetBitset() const { return bs_; }

  bool Empty() const { return bs_.empty(); }

  bool PartialEmpty(VID_T beg, VID_T end) const {
    return bs_.partial_empty(beg - beg_, end - beg_);
  }

 private:
  VID_T beg_;
  VID_T end_;
  Bitset bs_;
};

}  // namespace grape

#endif  // GRAPE_UTILS_VERTEX_SET_H_
