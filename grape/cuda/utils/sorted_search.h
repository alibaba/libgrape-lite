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

#ifndef GRAPE_CUDA_UTILS_SORTED_SEARCH_H_
#define GRAPE_CUDA_UTILS_SORTED_SEARCH_H_
#include <thrust/device_vector.h>

#include "grape/config.h"
#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/dev_utils.h"
#include "grape/cuda/utils/launcher.h"
#include "grape/cuda/utils/stream.h"

namespace grape {
namespace cuda {

template <typename T>
void sorted_search(const Stream& stream, T* needles, int num_needles,
                   T* haystack, int num_haystack, T* indices) {
  LaunchKernelFix(stream, num_haystack, [=] __device__() mutable {
    // assume num_needles > num_haystack
    if (num_needles == 0 || num_haystack == 0) {
      return;
    }
    int thread_lane = threadIdx.x & 31;  // thread index within the warp
    int warp_lane = threadIdx.x / 32;    // warp index within the CTA
    int nwarp = 256 / 32;
    __shared__ T cache[1024];
    int shm_size = 1024;
    int shm_per_warp = shm_size / nwarp;
    int shm_per_thd = shm_size / 256;
    T* my_cache = cache + warp_lane * shm_per_warp;
    size_t num = 0;
    for (int i = 0; i < shm_per_thd; ++i) {
      my_cache[i * 32 + thread_lane] =
          haystack[(thread_lane + i * 32) * num_haystack / shm_per_warp];
    }
    __syncwarp();

    for (auto i = thread_lane; i < num_needles; i += 32) {
      auto key = needles[i];
      auto idx = binary_search_2phase(haystack, my_cache, key, num_haystack,
                                      shm_per_warp);
      indices[i] = idx;
    }
  });

}  // namespace cuda
}  // namespace cuda

#endif  // GRAPE_CUDA_UTILS_SORTED_SEARCH_H_
