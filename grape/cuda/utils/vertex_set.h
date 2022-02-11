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

#ifndef GRAPE_CUDA_UTILS_VERTEX_SET_H_
#define GRAPE_CUDA_UTILS_VERTEX_SET_H_
#include "grape/config.h"
#include "grape/cuda/utils/bitset.h"
#include "grape/cuda/utils/vertex_array.h"

namespace grape {
namespace cuda {
namespace dev {
template <typename VID_T>
class DenseVertexSet {
 public:
  DenseVertexSet(VID_T beg, const Bitset<VID_T>& bitset)
      : beg_(beg), bitset_(bitset) {}

  DEV_INLINE bool Insert(Vertex<VID_T> v) {
    return bitset_.set_bit_atomic(v.GetValue() - beg_);
  }

  DEV_INLINE bool Exist(Vertex<VID_T> v) const {
    return bitset_.get_bit(v.GetValue() - beg_);
  }

  DEV_INLINE void Clear() { bitset_.clear(); }

  DEV_INLINE size_t Count() { return bitset_.get_positive_count(); }

 private:
  VID_T beg_;
  Bitset<VID_T> bitset_;
};
}  // namespace dev
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
      : beg_(range.begin_value()), end_(range.end_value()), bs_(end_ - beg_) {}

  ~DenseVertexSet() = default;

  void Init(const VertexRange<VID_T>& range) {
    beg_ = range.begin_value();
    end_ = range.end_value();
    bs_.Init(end_ - beg_);
    bs_.Clear();
  }

  dev::DenseVertexSet<VID_T> DeviceObject() {
    return dev::DenseVertexSet<VID_T>(beg_, bs_.DeviceObject());
  }

  void Insert(Vertex<VID_T> u) { bs_.SetBit(u.GetValue() - beg_); }

  VertexRange<VID_T> Range() const { return VertexRange<VID_T>(beg_, end_); }

  VID_T Count() const { return bs_.GetPositiveCount(); }

  VID_T Count(const Stream& stream) const {
    return bs_.GetPositiveCount(stream);
  }

  void Clear() { bs_.Clear(); }

  void Clear(const Stream& stream) { bs_.Clear(stream); }

  void Swap(DenseVertexSet<VID_T>& rhs) {
    std::swap(beg_, rhs.beg_);
    std::swap(end_, rhs.end_);
    bs_.Swap(rhs.bs_);
  }

 private:
  VID_T beg_{};
  VID_T end_{};
  Bitset<VID_T> bs_{};
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_VERTEX_SET_H_
