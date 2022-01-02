#ifndef GRAPE_GPU_UTILS_VERTEX_SET_H_
#define GRAPE_GPU_UTILS_VERTEX_SET_H_

#include "grape_gpu/utils/bitset.h"
#include "grape_gpu/utils/vertex_array.h"

namespace grape_gpu {

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
      : beg_(range.begin().GetValue()),
        end_(range.end().GetValue()),
        bs_(end_ - beg_) {}

  ~DenseVertexSet() = default;

  void Init(const VertexRange<VID_T>& range) {
    beg_ = range.begin().GetValue();
    end_ = range.end().GetValue();
    bs_.Init(end_ - beg_);
    bs_.Clear();
  }

  void Init(VID_T size) {
    beg_ = 0;
    end_ = size; 
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

/**
 * @brief A remote vertex set with dense vertices.
 *
 * @tparam VID_T Vertex ID type.
 */
template <typename VID_T>
class RemoteDenseVertexSet {
 public:
  RemoteDenseVertexSet() = default;

  void Init(const grape::CommSpec& comm_spec, const VertexRange<VID_T>& range) {
    beg_ = range.begin().GetValue();
    end_ = range.end().GetValue();
    bs_.Init(comm_spec, end_ - beg_);
    bs_.Clear();
  }
  
  dev::DenseVertexSet<VID_T> DeviceObject() {
    return dev::DenseVertexSet<VID_T>(beg_, bs_.DeviceObject());
  }

  dev::DenseVertexSet<VID_T> DeviceObject(int rid) {
    return dev::DenseVertexSet<VID_T>(beg_, bs_.DeviceObject(rid));
  }

  void Insert(Vertex<VID_T> u) { bs_.SetBit(u.GetValue() - beg_); }

  VertexRange<VID_T> Range() const { return VertexRange<VID_T>(beg_, end_); }

  VID_T Count() const { return bs_.GetPositiveCount(); }

  VID_T Count(const Stream& stream) const {
    return bs_.GetPositiveCount(stream);
  }

  void Clear() { bs_.Clear(); }

  void Swap(DenseVertexSet<VID_T>& rhs) {
    std::swap(beg_, rhs.beg_);
    std::swap(end_, rhs.end_);
    bs_.Swap(rhs.bs_);
  }

 private:
  VID_T beg_{};
  VID_T end_{};
  RemoteBitset<VID_T> bs_{};
};

}  // namespace grape_gpu
#endif  // GRAPE_GPU_UTILS_VERTEX_SET_H_
