#ifndef GRAPE_GPU_UTILS_CUDA_UTILS_H_
#define GRAPE_GPU_UTILS_CUDA_UTILS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector_types.h>

#include <cub/util_ptx.cuh>
#include <vector>

#include "glog/logging.h"

#ifdef __CUDACC__
#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#else
#define DEV_HOST_INLINE inline
#define DEV_HOST
#endif

#define MAX_BLOCK_SIZE 256
#define MAX_GRID_SIZE 768

__device__ static const char logtable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1,    0,     1,     1,     2,     2,     2,     2,     3,     3,     3,
    3,     3,     3,     3,     3,     LT(4), LT(5), LT(5), LT(6), LT(6), LT(6),
    LT(6), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)};

namespace grape_gpu {
#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)

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

// CUDA assertions
#define CHECK_CUDA(err)                                       \
  do {                                                        \
    cudaError_t errr = (err);                                 \
    if (errr != cudaSuccess) {                                \
      ::grape_gpu::HandleCudaError(__FILE__, __LINE__, errr); \
    }                                                         \
  } while (0)

#define CHECK_NCCL(err)                                       \
  do {                                                        \
    ncclResult_t errr = (err);                                \
    if (errr != ncclSuccess) {                                \
      ::grape_gpu::HandleNcclError(__FILE__, __LINE__, errr); \
    }                                                         \
  } while (0)

void ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm) {
  cudaError_t cudaErr;
  ncclResult_t ncclErr, ncclAsyncErr;
  while (true) {
    cudaErr = cudaStreamQuery(stream);
    if (cudaErr == cudaSuccess)
      return;

    if (cudaErr != cudaErrorNotReady) {
      LOG(FATAL) << "CUDA Error : cudaStreamQuery returned " << cudaErr;
    }

    ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
    if (ncclErr != ncclSuccess) {
      LOG(FATAL) << "NCCL Error : ncclCommGetAsyncError returned " << ncclErr;
    }

    if (ncclAsyncErr != ncclSuccess) {
      // An asynchronous error happened. Stop the operation and destroy
      // the communicator
      ncclErr = ncclCommAbort(comm);
      if (ncclErr != ncclSuccess) {
        LOG(FATAL) << "NCCL Error : ncclCommDestroy returned %d" << ncclErr;
      }
      // Caller may abort or try to re-create a new communicator.
      LOG(FATAL) << "ncclAsyncErr: " << ncclAsyncErr;
    }

    // We might want to let other threads (including NCCL threads) use the CPU.
    pthread_yield();
  }
}

}  // namespace grape_gpu

inline __host__ __device__ size_t round_up(size_t numerator,
                                           size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename SIZE_T>
inline void KernelSizing(int& block_num, int& block_size, SIZE_T work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num = std::min(MAX_GRID_SIZE, (int)round_up(work_size, block_size));
}

// Refer:https://github.com/gunrock/gunrock/blob/a7fc6948f397912ca0c8f1a8ccf27d1e9677f98f/gunrock/oprtr/intersection/cta.cuh#L84
__device__ unsigned ilog2(unsigned int v) {
  register unsigned int t, tt;
  if (tt = v >> 16)
    return ((t = tt >> 8) ? 24 + logtable[t] : 16 + logtable[tt]);
  else
    return ((t = v >> 8) ? 8 + logtable[t] : logtable[v]);
}
#endif  // GRAPE_GPU_UTILS_CUDA_UTILS_H_