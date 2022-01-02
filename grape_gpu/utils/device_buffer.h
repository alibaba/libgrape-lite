
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_DEVICE_BUFFER_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_DEVICE_BUFFER_H_
#include <cuda.h>

#include <memory>

#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
namespace grape_gpu {
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t size) { resize(size); }

  DeviceBuffer(const DeviceBuffer<T>& rhs) { *this = rhs; }

  DeviceBuffer(DeviceBuffer<T>&& rhs) noexcept { *this = rhs; }

  ~DeviceBuffer() { CHECK_CUDA(cudaFree(data_)); }

  DeviceBuffer& operator=(const DeviceBuffer<T>& rhs) {
    if (&rhs != this) {
      resize(rhs.size_);
    }
    return *this;
  }

  DeviceBuffer& operator=(DeviceBuffer<T>&& rhs) noexcept {
    if (&rhs != this) {
      CHECK_CUDA(cudaFree(data_));
      data_ = rhs.data_;
      size_ = rhs.size_;
      capacity_ = rhs.capacity_;
      rhs.data_ = nullptr;
      rhs.size_ = 0;
      rhs.capacity_ = 0;
    }
    return *this;
  }

  void resize(size_t size, bool keep_data = true) {
    if (size > capacity_) {
      if (keep_data) {
        T* data;
        CHECK_CUDA(cudaMalloc((void**) &data, sizeof(T) * size));
        CHECK_CUDA(cudaMemcpy(data, data_, size_ * sizeof(T),
                              cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaFree(data_));
        data_ = data;
      } else {
        CHECK_CUDA(cudaFree(data_));
        CHECK_CUDA(cudaMalloc((void**) &data_, sizeof(T) * size));
      }
      capacity_ = size;
    }
    size_ = size;
  }

  T* data() { return data_; }

  const T* data() const { return data_; }

  size_t size() const { return size_; }

  ArrayView<T> DeviceObject() { return ArrayView<T>(data_, size_); }

 private:
  size_t capacity_{};
  size_t size_{};
  T* data_{};
};
}  // namespace grape_gpu

#endif  // GRAPEGPU_GRAPE_GPU_UTILS_DEVICE_BUFFER_H_
