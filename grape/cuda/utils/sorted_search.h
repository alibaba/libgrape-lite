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
#include "grape/cuda/utils/launcher.h"
#include "moderngpu/kernel_sortedsearch.hxx"

namespace grape {
namespace cuda {
template <mgpu::bounds_t bounds, typename a_keys_it, typename b_keys_it,
          typename comp_t>
void merge_path_partitions(const Stream& stream,
                           thrust::device_vector<int>& partitions, a_keys_it a,
                           int64_t a_count, b_keys_it b, int64_t b_count,
                           int64_t spacing, comp_t comp) {
  typedef int int_t;
  auto num_partitions = dev::round_up(a_count + b_count, spacing) + 1;

  partitions.resize(num_partitions);

  KernelWrapper<<<1, 128, 0, stream.cuda_stream()>>>(
      [=] __device__(int* p) {
        auto tid = TID_1D;
        auto nthreads = TOTAL_THREADS_1D;

        for (int index = 0 + tid; index < num_partitions; index += nthreads) {
          auto diag = (int_t) min(spacing * index, a_count + b_count);
          p[index] = mgpu::merge_path<bounds>(a, (int_t) a_count, b,
                                              (int_t) b_count, diag, comp);
        }
      },
      thrust::raw_pointer_cast(partitions.data()));
}

template <mgpu::bounds_t bounds, typename needles_it, typename haystack_it,
          typename indices_it, typename comp_it>
void sorted_search(const Stream& stream, needles_it needles, int num_needles,
                   haystack_it haystack, int num_haystack, indices_it indices,
                   comp_it comp) {
  const int nt = 128;  // block size
  const int vt = 11;   // number of work-items per thread
  const int nv = nt * vt;

  typedef typename std::iterator_traits<needles_it>::value_type type_t;

  thrust::device_vector<int> partitions;
  // Partition the needles and haystacks into tiles.
  merge_path_partitions<bounds>(stream, partitions, needles, num_needles,
                                haystack, num_haystack, nv, comp);

  size_t num_partitions = partitions.size();
  const int* mp_data = thrust::raw_pointer_cast(partitions.data());
  auto size = dev::round_up(num_needles + num_haystack, nv);
  dim3 grid_dims(size, 1, 1), block_dims(nt, 1, 1);

  KernelWrapper<<<grid_dims, block_dims, 0, stream.cuda_stream()>>>(
      [=] __device__() {
        int tid = static_cast<int>(threadIdx.x % (unsigned) nt);
        int cta = blockIdx.x;

        __shared__ union {
          type_t keys[nv + 1];  // NOLINT(runtime/arrays)
          int indices[nv];      // NOLINT(runtime/arrays)
        } shared;

        // Load the range for this CTA and merge the values into register.
        int mp0 = mp_data[cta + 0];
        int mp1 = mp_data[cta + 1];
        mgpu::merge_range_t range = mgpu::compute_merge_range(
            num_needles, num_haystack, cta, nv, mp0, mp1);

        // Merge the values needles and haystack.
        mgpu::merge_pair_t<type_t, vt> merge =
            mgpu::cta_merge_from_mem<bounds, nt, vt>(needles, haystack, range,
                                                     tid, comp, shared.keys);

        // Store the needle indices to shared memory.
        mgpu::iterate<vt>([&](int i) {
          if (merge.indices[i] < range.a_count()) {
            int needle = merge.indices[i];
            int haystack = range.b_begin + vt * tid + i - needle;
            shared.indices[needle] = haystack;
          }
        });
        __syncthreads();

        mgpu::shared_to_mem<nt, vt>(shared.indices, tid, range.a_count(),
                                    indices + range.a_begin);
      });
}
}  // namespace cuda

}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_SORTED_SEARCH_H_
