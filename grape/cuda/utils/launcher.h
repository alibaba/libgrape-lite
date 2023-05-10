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

#ifndef GRAPE_CUDA_UTILS_LAUNCHER_H_
#define GRAPE_CUDA_UTILS_LAUNCHER_H_
#include "grape/cuda/utils/stream.h"

namespace grape {

namespace cuda {
template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
      reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernelFix(const Stream& stream, size_t size, F f, Args&&... args) {
  KernelWrapper<<<256, 256, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_LAUNCHER_H_
