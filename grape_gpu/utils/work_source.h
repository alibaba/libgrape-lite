
#ifndef GRAPE_GPU_UTILS_WORK_SOURCE_H_
#define GRAPE_GPU_UTILS_WORK_SOURCE_H_
#include "grape_gpu/utils/cuda_utils.h"
namespace grape_gpu {
template <typename T>
struct WorkSourceRange {
 public:
  WorkSourceRange() = default;

  DEV_HOST WorkSourceRange(T start, size_t size) : start_(start), size_(size) {}

  DEV_HOST_INLINE T GetWork(size_t i) const { return (T) (start_ + i); }

  DEV_HOST_INLINE size_t size() const { return size_; }

 private:
  T start_{};
  size_t size_{};
};

template <typename T>
struct WorkSourceArray {
 public:
  WorkSourceArray() = default;

  DEV_HOST WorkSourceArray(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T GetWork(size_t i) const { return data_[i]; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE T* data() const { return data_; }

 private:
  T* data_{};
  size_t size_{};
};

}  // namespace grape_gpu

#endif  // GRAPE_GPU_UTILS_WORK_SOURCE_H_
