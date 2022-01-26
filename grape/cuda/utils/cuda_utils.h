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

#ifndef GRAPE_CUDA_UTILS_CUDA_UTILS_H_
#define GRAPE_CUDA_UTILS_CUDA_UTILS_H_
#include <cuda.h>
#include <nccl.h>

#include "grape/config.h"

namespace grape {
namespace cuda {
static void HandleCudaError(const char* file, int line, cudaError_t err) {
  LOG(FATAL) << "ERROR in " << file << ":" << line << ": "
             << cudaGetErrorString(err) << " (" << err << ")";
}

static void HandleNcclError(const char* file, int line, ncclResult_t err) {
  std::string error_msg;
  switch (err) {
  case ncclUnhandledCudaError:
    error_msg = "ncclUnhandledCudaError";
    break;
  case ncclSystemError:
    error_msg = "ncclSystemError";
    break;
  case ncclInternalError:
    error_msg = "ncclInternalError";
    break;
  case ncclInvalidArgument:
    error_msg = "ncclInvalidArgument";
    break;
  case ncclInvalidUsage:
    error_msg = "ncclInvalidUsage";
    break;
  case ncclNumResults:
    error_msg = "ncclNumResults";
    break;
  default:
    error_msg = "";
  }
  LOG(FATAL) << "ERROR in " << file << ":" << line << ": " << error_msg;
}

inline void KernelSizing(int& block_num, int& block_size, size_t work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num =
      std::min(MAX_GRID_SIZE,
               static_cast<int>((work_size + block_size - 1) / block_size));
}

}  // namespace cuda
}  // namespace grape

#define CHECK_CUDA(err)                                       \
  do {                                                        \
    cudaError_t errr = (err);                                 \
    if (errr != cudaSuccess) {                                \
      grape::cuda::HandleCudaError(__FILE__, __LINE__, errr); \
    }                                                         \
  } while (0)

#define CHECK_NCCL(err)                                       \
  do {                                                        \
    ncclResult_t errr = (err);                                \
    if (errr != ncclSuccess) {                                \
      grape::cuda::HandleNcclError(__FILE__, __LINE__, errr); \
    }                                                         \
  } while (0)
#endif  // GRAPE_CUDA_UTILS_CUDA_UTILS_H_
