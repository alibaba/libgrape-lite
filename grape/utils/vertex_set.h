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
#include "grape/utils/vertex_array.h"

namespace grape {
/**
 * @brief A vertex set with dense vertices.
 *
 * @tparam VID_T Vertex ID type.
 */
template <typename VID_T>
class DenseVertexSet {
 public:
  DenseVertexSet() = default;

  explicit DenseVertexSet(const VertexRange<VID_T>& range)
      : beg_(range.begin().GetValue()),
        end_(range.end().GetValue()),
        bs_(end_ - beg_) {}

  ~DenseVertexSet() = default;

  void Init(const VertexRange<VID_T>& range, int thread_num = 1) {
    beg_ = range.begin().GetValue();
    end_ = range.end().GetValue();
    bs_.init(end_ - beg_);
    if (thread_num == 1) {
      bs_.clear();
    } else {
      bs_.parallel_clear(thread_num);
    }
  }

  void Init(const VertexVector<VID_T>& vertices, int thread_num = 1) {
    if (vertices.size() == 0)
      return;
    beg_ = vertices[0].GetValue();
    end_ = vertices[vertices.size() - 1].GetValue();
    bs_.init(end_ - beg_ + 1);
    if (thread_num == 1) {
      bs_.clear();
    } else {
      bs_.parallel_clear(thread_num);
    }
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

  size_t ParallelCount(int thread_num) const {
    return bs_.parallel_count(thread_num);
  }

  size_t PartialCount(VID_T beg, VID_T end) const {
    return bs_.partial_count(beg - beg_, end - beg_);
  }

  size_t ParallelPartialCount(int thread_num, VID_T beg, VID_T end) const {
    return bs_.parallel_partial_count(thread_num, beg - beg_, end - beg_);
  }

  void Clear() { bs_.clear(); }

  void ParallelClear(int thread_num) { bs_.parallel_clear(thread_num); }

  void Swap(DenseVertexSet<VID_T>& rhs) {
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
