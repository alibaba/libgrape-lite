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
  KernelWrapper<<<256, 256, 0, stream.cuda_stream()>>>(
      [=] __device__() mutable {
        auto nthreads = gridDim.x * blockDim.x;
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        // assume num_needles > num_haystack
        if (num_needles == 0 || num_haystack == 0) {
          return;
        }

        for (auto i = tid; i < num_needles; i += nthreads) {
          auto key = needles[i];
          int s = 0;
          int len = num_haystack;
          while (len > 0) {
            int half = len >> 1;
            int mid = s + half;
            if (haystack[mid] < key) {
              s = mid + 1;
              len = len - half - 1;
            } else {
              len = half;
            }
          }
          indices[i] = s;
        }
      });
}

}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_SORTED_SEARCH_H_
