
#ifndef GRAPE_GPU_UTILS_DEV_VERTEX_ARRAY_H_
#define GRAPE_GPU_UTILS_DEV_VERTEX_ARRAY_H_
#include <thrust/device_vector.h>

#include <cub/util_type.cuh>

#include "grape/utils/gcontainer.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/ipc_array.h"
#include "grape_gpu/utils/stream.h"

namespace grape_gpu {

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
  DEV_HOST explicit Vertex(T value) : value_(value) {}
  DEV_HOST Vertex(const Vertex& rhs) : value_(rhs.value_) {}
  DEV_HOST Vertex(Vertex&& rhs) noexcept : value_(rhs.value_) {}

  DEV_HOST_INLINE Vertex& operator=(const Vertex& rhs) {
    value_ = rhs.value_;
    return *this;
  }

  DEV_HOST_INLINE Vertex& operator=(Vertex&& rhs) noexcept {
    value_ = rhs.value_;
    return *this;
  }

  DEV_HOST_INLINE Vertex& operator=(T value) {
    value_ = value;
    return *this;
  }

  DEV_HOST_INLINE Vertex& operator++() {
    value_++;
    return *this;
  }

  DEV_HOST_INLINE Vertex operator++(int) {
    Vertex res(value_);
    value_++;
    return res;
  }

  DEV_HOST_INLINE Vertex& operator--() {
    value_--;
    return *this;
  }

  DEV_HOST_INLINE Vertex operator--(int) {
    Vertex res(value_);
    value_--;
    return res;
  }

  DEV_HOST_INLINE Vertex operator+(T inc) const { return Vertex(value_ + inc); }

  DEV_HOST_INLINE bool operator==(const Vertex& rhs) const {
    return value_ == rhs.value_;
  }

  DEV_HOST_INLINE bool operator!=(const Vertex& rhs) const {
    return value_ != rhs.value_;
  }

  void Swap(Vertex& rhs) { std::swap(value_, rhs.value_); }

  DEV_HOST_INLINE bool operator<(const Vertex& rhs) const {
    return value_ < rhs.value_;
  }

  DEV_HOST_INLINE bool operator>(const Vertex& rhs) const {
    return value_ > rhs.value_;
  }

  DEV_HOST_INLINE Vertex& operator*() { return *this; }

  DEV_HOST_INLINE T GetValue() const { return value_; }

  DEV_HOST_INLINE void SetValue(T value) { value_ = value; }

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

template <typename VID_T>
grape::InArchive& operator<<(grape::InArchive& archive,
                             const Vertex<VID_T>& v) {
  archive << v.GetValue();
  return archive;
}

template <typename VID_T>
grape::OutArchive& operator>>(grape::OutArchive& archive, Vertex<VID_T>& v) {
  VID_T vid;
  archive >> vid;
  v.SetValue(vid);
  return archive;
}

template <typename T>
class VertexRange {
 public:
  VertexRange() = default;
  DEV_HOST VertexRange(T begin, T end)
      : begin_(begin), end_(end), size_(end - begin) {}
  DEV_HOST VertexRange(const Vertex<T>& begin, const Vertex<T>& end)
      : begin_(begin), end_(end), size_(end.GetValue() - begin.GetValue()) {}
  DEV_HOST VertexRange(const VertexRange& r)
      : begin_(r.begin_), end_(r.end_), size_(r.size_) {}

  DEV_HOST_INLINE const Vertex<T>& begin() const { return begin_; }

  DEV_HOST_INLINE const Vertex<T>& end() const { return end_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

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
  size_t size_{};
};

namespace dev {
template <typename T, typename VID_T>
class VertexArray {
 public:
  VertexArray() = default;

  DEV_HOST VertexArray(VertexRange<VID_T> range, T* data)
      : range_(range),
        data_(data),
        fake_start_(data - range.begin().GetValue()) {}

  DEV_INLINE T& operator[](const Vertex<VID_T>& loc) {
    return fake_start_[loc.GetValue()];
  }

  DEV_INLINE const T& operator[](const Vertex<VID_T>& loc) const {
    return fake_start_[loc.GetValue()];
  }

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE size_t size() const { return range_.size(); }

 private:
  VertexRange<VID_T> range_;
  T* data_{};
  T* fake_start_{};
};

}  // namespace dev

template <typename T, typename VID_T>
class VertexArray : public grape::Array<T, grape::Allocator<T>> {
  using Base = grape::Array<T, grape::Allocator<T>>;

 public:
  VertexArray() : Base(), fake_start_(NULL) {}

  explicit VertexArray(const VertexRange<VID_T>& range)
      : Base(range.size()), range_(range) {
    fake_start_ = Base::data() - range_.begin().GetValue();
    d_data_.resize(range.size());
  }

  VertexArray(const VertexRange<VID_T>& range, const T& value)
      : Base(range.size(), value), range_(range) {
    fake_start_ = Base::data() - range_.begin().GetValue();
    d_data_.resize(range.size());
  }

  ~VertexArray() = default;

  void Init(const VertexRange<VID_T>& range) {
    Base::clear();
    Base::resize(range.size());
    range_ = range;
    fake_start_ = Base::data() - range_.begin().GetValue();
    d_data_.clear();
    d_data_.resize(range.size());
  }

  void Init(const VertexRange<VID_T>& range, const T& value) {
    Base::clear();
    Base::resize(range.size(), value);
    range_ = range;
    fake_start_ = Base::data() - range_.begin().GetValue();
    d_data_.clear();
    d_data_.resize(range.size(), value);
  }

  void SetValue(VertexRange<VID_T>& range, const T& value) {
    std::fill_n(
        &Base::data()[range.begin().GetValue() - range_.begin().GetValue()],
        range.size(), value);
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](Vertex<VID_T>& loc) {
    return fake_start_[loc.GetValue()];
  }

  inline const T& operator[](const Vertex<VID_T>& loc) const {
    return fake_start_[loc.GetValue()];
  }

  void resize(size_t size) {
    grape::Array<T, grape::Allocator<T>>::resize(size);
    d_data_.resize(size);
    d_data_.shrink_to_fit();
  }

  void Swap(VertexArray& rhs) {
    Base::swap((Base&) rhs);
    range_.Swap(rhs.range_);
    std::swap(fake_start_, rhs.fake_start_);
    d_data_.swap(rhs.d_data_);
  }

  void Clear() {
    VertexArray ga;
    this->Swap(ga);
  }

  const VertexRange<VID_T>& GetVertexRange() const { return range_; }

  dev::VertexArray<T, VID_T> DeviceObject() {
    return dev::VertexArray<T, VID_T>(range_,
                                      thrust::raw_pointer_cast(d_data_.data()));
  }

  void H2D() {
    CHECK_CUDA(cudaMemcpy(d_data_.data().get(), this->data(),
                          sizeof(T) * this->size(), cudaMemcpyHostToDevice));
  }

  void H2D(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(d_data_.data().get(), this->data(),
                               sizeof(T) * this->size(), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  void D2H() {
    CHECK_CUDA(cudaMemcpy(this->data(), d_data_.data().get(),
                          sizeof(T) * this->size(), cudaMemcpyDeviceToHost));
  }

  void D2H(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(this->data(), d_data_.data().get(),
                               sizeof(T) * this->size(), cudaMemcpyDeviceToHost,
                               stream.cuda_stream()));
  }

 private:
  VertexRange<VID_T> range_;
  T* fake_start_;
  thrust::device_vector<T> d_data_;
};

template <typename T, typename VID_T>
class RemoteVertexArray : public grape::Array<T, grape::Allocator<T>> {
  using Base = grape::Array<T, grape::Allocator<T>>;

 public:
  RemoteVertexArray() : Base(), fake_start_(NULL) {}

  ~RemoteVertexArray() = default;

  void Init(const grape::CommSpec& comm_spec, const VertexRange<VID_T>& range) {
    Base::clear();
    Base::resize(range.size());

    range_ = std::move(
        IPCArray<VertexRange<VID_T>, IPCMemoryPlacement::kHost>(comm_spec));
    range_.Init(1, range);
    fake_start_ = Base::data() - range.begin().GetValue();
    d_data_ = std::move(IPCArray<T, IPCMemoryPlacement::kDevice>(comm_spec));
    d_data_.Init(range.size());
  }

  void Init(const grape::CommSpec& comm_spec, const VertexRange<VID_T>& range,
            const T& value) {
    Base::clear();
    Base::resize(range.size(), value);

    range_ = std::move(
        IPCArray<VertexRange<VID_T>, IPCMemoryPlacement::kHost>(comm_spec));
    range_.Init(1, range);
    fake_start_ = Base::data() - range.begin().GetValue();
    d_data_ = std::move(IPCArray<T, IPCMemoryPlacement::kDevice>(comm_spec));
    d_data_.Init(range.size(), value);
  }

  void SetValue(VertexRange<VID_T>& range, const T& value) {
    auto base_range = *range_.local_view().data();

    std::fill_n(
        &Base::data()[range.begin().GetValue() - base_range.begin().GetValue()],
        range.size(), value);
  }

  void SetValue(const T& value) {
    std::fill_n(Base::data(), Base::size(), value);
  }

  inline T& operator[](Vertex<VID_T>& loc) {
    return fake_start_[loc.GetValue()];
  }

  inline const T& operator[](const Vertex<VID_T>& loc) const {
    return fake_start_[loc.GetValue()];
  }

  void Swap(RemoteVertexArray& rhs) {
    Base::swap((Base&) rhs);
    range_.Swap(rhs.range_);
    std::swap(fake_start_, rhs.fake_start_);
    d_data_.Swap(rhs.d_data_);
  }

  void Clear() {
    RemoteVertexArray ga;
    this->Swap(ga);
  }

  const VertexRange<VID_T>& GetVertexRange() const {
    return range_.local_view()[0];
  }

  dev::VertexArray<T, VID_T> DeviceObject() {
    return dev::VertexArray<T, VID_T>(range_.local_view()[0],
                                      d_data_.local_view().data());
  }

  dev::VertexArray<T, VID_T> DeviceObject(int local_id) {
    return dev::VertexArray<T, VID_T>(range_.view(local_id)[0],
                                      d_data_.view(local_id).data());
  }

  void H2D() {
    CHECK_CUDA(cudaMemcpy(d_data_.local_view().data(), this->data(),
                          sizeof(T) * this->size(), cudaMemcpyHostToDevice));
  }

  void H2D(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(d_data_.local_view().data(), this->data(),
                               sizeof(T) * this->size(), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  void D2H() {
    CHECK_CUDA(cudaMemcpy(this->data(), d_data_.local_view().data(),
                          sizeof(T) * this->size(), cudaMemcpyDeviceToHost));
  }

  void D2H(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(this->data(), d_data_.local_view().data(),
                               sizeof(T) * this->size(), cudaMemcpyDeviceToHost,
                               stream.cuda_stream()));
  }

 private:
  IPCArray<VertexRange<VID_T>, IPCMemoryPlacement::kHost> range_;
  T* fake_start_;

  IPCArray<T, IPCMemoryPlacement::kDevice> d_data_;
};

}  // namespace grape_gpu

namespace cub {
// Making Vertex to be comparable for cub's segmented radix sort
template <typename T>
struct Traits<grape_gpu::Vertex<T>>
    : NumericTraits<typename RemoveQualifiers<T>::Type> {};

}  // namespace cub

namespace std {
template <typename VID_T>
struct hash<grape_gpu::Vertex<VID_T>> {
  std::size_t operator()(const grape_gpu::Vertex<VID_T>& v) const {
    return v.GetValue();
  }
};
}  // namespace std

#endif  // GRAPE_GPU_UTILS_DEV_VERTEX_ARRAY_H_
