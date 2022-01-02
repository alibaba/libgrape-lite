#ifndef GRAPE_GPU_GPU_CONFIG_H_
#define GRAPE_GPU_GPU_CONFIG_H_
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "grape/config.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape_gpu {
using fid_t = grape::fid_t;
template <typename T>
using pinned_vector =
    thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;

}  // namespace grape_gpu

namespace grape {
template <typename T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline grape::InArchive& operator<<(grape::InArchive& in_archive,
                                    const grape_gpu::pinned_vector<T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  in_archive.AddBytes(vec.data(), size * sizeof(T));
  return in_archive;
}

template <typename T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline grape::OutArchive& operator>>(grape::OutArchive& out_archive,
                                     grape_gpu::pinned_vector<T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  memcpy(&vec[0], out_archive.GetBytes(sizeof(T) * size), sizeof(T) * size);
  return out_archive;
}
}  // namespace grape

#endif  // GRAPE_GPU_CONFIG_H_
