
#ifndef GRAPE_GPU_UTILS_SHARED_VALUE_H_
#define GRAPE_GPU_UTILS_SHARED_VALUE_H_
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/ipc_array.h"

namespace grape_gpu {

template <typename T>
class SharedValue {
  static_assert(std::is_pod<T>::value, "Unsupported datatype");

 public:
  SharedValue() {
    d_buffer_.resize(1);
    h_buffer_.resize(1);
  }

  void set(const T& t) { d_buffer_[0] = t; }

  void set(const T& t, const Stream& stream) {
    h_buffer_[0] = t;
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  typename thrust::device_vector<T>::reference get() { return d_buffer_[0]; }

  typename thrust::device_vector<T>::const_reference get() const {
    return d_buffer_[0];
  }

  T get(const Stream& stream) const {
    CHECK_CUDA(
        cudaMemcpyAsync((void*) thrust::raw_pointer_cast(h_buffer_.data()),
                        thrust::raw_pointer_cast(d_buffer_.data()), sizeof(T),
                        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_[0];
  }

  T* data() { return thrust::raw_pointer_cast(d_buffer_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(d_buffer_.data()); }

  void Assign(const SharedValue<T>& rhs) {
    CHECK_CUDA(cudaMemcpy(thrust::raw_pointer_cast(d_buffer_.data()),
                          thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Assign(const SharedValue<T>& rhs, const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                               sizeof(T), cudaMemcpyDefault,
                               stream.cuda_stream()));
  }

  void Swap(SharedValue<T>& rhs) {
    d_buffer_.swap(rhs.d_buffer_);
    h_buffer_.swap(rhs.h_buffer_);
  }

 private:
  thrust::device_vector<T> d_buffer_;
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>
      h_buffer_;
};

template <typename T>
class RemoteSharedValue {
  static_assert(std::is_pod<T>::value, "Unsupported datatype");

 public:
  RemoteSharedValue() {}

  explicit RemoteSharedValue(const grape::CommSpec& comm_spec) 
      : d_buffers_(comm_spec) {
    h_buffers_.resize(1);
    d_buffers_.Init(1);
  }

  void Init() {

  }

  RemoteSharedValue(RemoteSharedValue&& rhs) noexcept {
    *this = std::move(rhs);
  }

  RemoteSharedValue& operator=(RemoteSharedValue&& rhs) noexcept {
    if(this != &rhs) {
      d_buffers_ = std::move(rhs.d_buffers_);
      h_buffers_ = std::move(rhs.h_buffers_);
    }
    return *this;
  }

  void set(const T& t) { 
    h_buffers_[0] = t;
    CHECK_CUDA(cudaMemcpy(d_buffers_.local_view().data(),
                          thrust::raw_pointer_cast(h_buffers_.data()),
                          sizeof(T), cudaMemcpyHostToDevice));
  }

  void set(const T& t, const Stream& stream) {
    h_buffers_[0] = t;
    CHECK_CUDA(cudaMemcpyAsync(d_buffers_.local_view().data(),
                               thrust::raw_pointer_cast(h_buffers_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }
  
  void set(int rid, const T& t, const Stream& stream) {
    h_buffers_[0] = t;
    CHECK_CUDA(cudaMemcpyAsync(d_buffers_.view(rid).data(),
                               thrust::raw_pointer_cast(h_buffers_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  //typename thrust::device_vector<T>::reference get() { return d_buffer_[0]; }

  //typename thrust::device_vector<T>::const_reference get() const {
  //  return d_buffer_[0];
  //}
  
  T get() {
    CHECK_CUDA(
        cudaMemcpy((void*) thrust::raw_pointer_cast(h_buffers_.data()),
                           d_buffers_.local_view().data(), sizeof(T),
                           cudaMemcpyDeviceToHost));
    return h_buffers_[0];
  }

  T get(const Stream& stream) {
    CHECK_CUDA(
        cudaMemcpyAsync((void*) thrust::raw_pointer_cast(h_buffers_.data()),
                        d_buffers_.local_view().data(), sizeof(T),
                        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffers_[0];
  }

  T get(int rid, const Stream& stream) {
    CHECK_CUDA(
        cudaMemcpyAsync((void*) thrust::raw_pointer_cast(h_buffers_.data()),
                        d_buffers_.view(rid).data(), sizeof(T),
                        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffers_[0];
  }

  T* data() { return d_buffers_.local_view().data(); }

  const T* data() const { return d_buffers_.local_view().data(); }

  T* data(int rid) { return d_buffers_.view(rid).data(); }

  const T* data(int rid) const { return d_buffers_.view(rid).data(); }

  void Assign(const RemoteSharedValue<T>& rhs) {
    CHECK_CUDA(cudaMemcpy(d_buffers_.local_view().data(),
                          rhs.d_buffers_.local_view().data(),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Assign(const RemoteSharedValue<T>& rhs, const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(d_buffers_.local_view().data(),
                               rhs.d_buffers_.local_view().data(),
                               sizeof(T), cudaMemcpyDefault,
                               stream.cuda_stream()));
  }

  void Swap(RemoteSharedValue<T>& rhs) {
    d_buffers_.Swap(rhs.d_buffers_);
    h_buffers_.swap(rhs.h_buffers_);
  }

 private:
  //thrust::device_vector<T> d_buffer_;
  IPCArray<T, IPCMemoryPlacement::kDevice> d_buffers_;
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>
      h_buffers_;
};
}  // namespace grape_gpu
#endif  // GRAPE_GPU_UTILS_SHARED_VALUE_H_
