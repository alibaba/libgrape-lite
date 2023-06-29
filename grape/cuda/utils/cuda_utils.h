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
#include <sys/resource.h>
#include <sys/time.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "cub/cub.cuh"
#include "grape/config.h"

#if defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <fcntl.h>
#include <malloc.h>
#include <sys/statvfs.h>
#endif
#endif

#ifdef __linux__
#ifndef SHMMAX_SYS_FILE
#define SHMMAX_SYS_FILE "/proc/sys/kernel/shmmax"
#endif
#else
#include <sys/sysctl.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>

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

inline void ReportMemoryUsage(std::string marker) {
  size_t free_byte, total_byte;
  double giga = (1ul) << 30;
  cudaError_t error_id = cudaMemGetInfo(&free_byte, &total_byte);
  if (error_id != cudaSuccess) {
    printf("cudaMemGetInfo() failed: %s\n", cudaGetErrorString(error_id));
    return;
  }
  printf("%s: Global Memory[Total, Free] (Gb): [%.3f, %.3f]\n", marker.c_str(),
         total_byte / giga, free_byte / giga);
}

void trim_rss() {
#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  malloc_trim(1024 * 1024 /* 1MB */);
#endif
}

size_t get_rss(bool include_shared_memory) {
  // why "trim_rss" first?
  //
  //  - for more accurate statistics
  //  - as a hint for allocator to release pages in places where `get_rss()`
  //    is called (where memory information is in cencern) in programs.
  trim_rss();

#if defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info,
                &infoCount) != KERN_SUCCESS)
    return (size_t) 0L; /* Can't access? */
  return (size_t) info.resident_size;
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  // /* Linux ---------------------------------------------------- */
  int64_t rss = 0L, shared_rss = 0L;
  FILE* fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t) 0L; /* Can't open? */
  //
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t) 0L; /* Can't read? */
  }
  // read the second number
  if (fscanf(fp, "%ld", &shared_rss) != 1) {
    fclose(fp);
    return (size_t) 0L; /* Can't read? */
  }
  fclose(fp);
  if (include_shared_memory) {
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
  } else {
    return (size_t)(rss - shared_rss) * (size_t) sysconf(_SC_PAGESIZE);
  }
#else
  /* Unknown OS ----------------------------------------------- */
  return 0;
#endif
}

inline void ReportHostMemoryUsage(std::string marker) {
  struct rusage usage;
  double giga = (1ul) << 30;
  size_t bytes = get_rss(false);
  if (bytes == 0) {
    getrusage(RUSAGE_SELF, &usage);
    bytes = (size_t) usage.ru_maxrss * 1024;
  }
  printf("%s: Host Memory[Used] (Gb): %.3f\n", marker.c_str(), bytes / giga);
}

template <typename KeyT, typename OffsetIteratorT>
static cudaError_t SortKeys64(void* d_temp_storage, size_t& temp_storage_bytes,
                              cub::DoubleBuffer<KeyT>& d_keys,
                              int64_t num_items, int64_t num_segments,
                              OffsetIteratorT d_begin_offsets,
                              OffsetIteratorT d_end_offsets, int begin_bit = 0,
                              int end_bit = sizeof(KeyT) * 8,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false) {
  // Signed integer type for global offsets
  typedef int64_t OffsetT;

  // Null value type
  cub::DoubleBuffer<cub::NullType> d_values;
#if CUB_VERSION > 200000
  return cub::DispatchSegmentedRadixSort<
      false, KeyT, cub::NullType, OffsetIteratorT, OffsetIteratorT,
      OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_values,
                         num_items, num_segments, d_begin_offsets,
                         d_end_offsets, begin_bit, end_bit, true, stream);
#else
  return cub::DispatchSegmentedRadixSort<
      false, KeyT, cub::NullType, OffsetIteratorT,
      OffsetT>::Dispatch(d_temp_storage, temp_storage_bytes, d_keys, d_values,
                         num_items, num_segments, d_begin_offsets,
                         d_end_offsets, begin_bit, end_bit, true, stream,
                         debug_synchronous);
#endif
}

template <typename InputIteratorT, typename OutputIteratorT>
static cudaError_t PrefixSumKernel64(void* d_temp_storage,
                                     size_t& temp_storage_bytes,
                                     InputIteratorT d_in, OutputIteratorT d_out,
                                     int num_items, cudaStream_t stream = 0,
                                     bool debug_synchronous = false) {
  // Signed integer type for global offsets
  typedef int OffsetT;

  // use size_t for aggregated value.
#if CUB_VERSION > 200000
  struct InitValue {
    using value_type = size_t;
  };
  return cub::DispatchScan<InputIteratorT, OutputIteratorT, cub::Sum, InitValue,
                           OffsetT>::Dispatch(d_temp_storage,
                                              temp_storage_bytes, d_in, d_out,
                                              cub::Sum(), 0, num_items, stream);
#else
  return cub::DispatchScan<InputIteratorT, OutputIteratorT, cub::Sum, size_t,
                           OffsetT>::Dispatch(d_temp_storage,
                                              temp_storage_bytes, d_in, d_out,
                                              cub::Sum(), 0, num_items, stream,
                                              debug_synchronous);
#endif
}

template <typename T>
T* SegmentSort(T* d_keys_in, T* d_keys_buffer, size_t* d_offset_lo,
               size_t* d_offset_hi, size_t num_items, size_t num_segments) {
  if (num_items <= 0 || num_segments <= 0)
    return d_keys_in;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  const size_t MAX_SIZE = (1ul << 31) - 1ul;

  if (num_items <= MAX_SIZE) {
    cub::DoubleBuffer<T> d_keys(d_keys_in, d_keys_buffer);
    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
        d_offset_lo, d_offset_hi));
    // Allocate temporary storage
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run sorting operation
    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_keys, num_items, num_segments,
        d_offset_lo, d_offset_hi));
    CHECK_CUDA(cudaFree(d_temp_storage));
    return d_keys.Current();
  } else {
    cub::DoubleBuffer<T> d_keys(d_keys_in, d_keys_buffer);
    CHECK_CUDA(SortKeys64(d_temp_storage, temp_storage_bytes, d_keys,
                          (int64_t) num_items, (int64_t) num_segments,
                          d_offset_lo, d_offset_hi));
    // Allocate temporary storage
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // std::cout << temp_storage_bytes << std::endl;
    // Run sorting operation
    CHECK_CUDA(SortKeys64(d_temp_storage, temp_storage_bytes, d_keys,
                          (int64_t) num_items, (int64_t) num_segments,
                          d_offset_lo, d_offset_hi));
    CHECK_CUDA(cudaFree(d_temp_storage));
    return d_keys.Current();
  }
}

template <typename I, typename O>
// Force the accumulate type as size_t, and it will always be ExclusiveSum
void ExclusiveSum64(I* d_keys_in, O* d_keys_out, size_t size,
                    cudaStream_t d_stream) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  const size_t MAX_SIZE = (1ul << 31) - 1ul;
  assert(size <= MAX_SIZE);

  CHECK_CUDA(PrefixSumKernel64(d_temp_storage, temp_storage_bytes, d_keys_in,
                               d_keys_out, size, d_stream));
  CHECK_CUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, d_stream));
  CHECK_CUDA(PrefixSumKernel64(d_temp_storage, temp_storage_bytes, d_keys_in,
                               d_keys_out, size, d_stream));
  CHECK_CUDA(cudaFreeAsync(d_temp_storage, d_stream));
}

template <typename I, typename O>
// The accumulate type is I;
void InclusiveSum(I* d_keys_in, O* d_keys_out, size_t size,
                  cudaStream_t d_stream) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  const size_t MAX_SIZE = (1ul << 31) - 1ul;
  assert(size <= MAX_SIZE);

  CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, size,
                                           d_stream));
  CHECK_CUDA(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, d_stream));
  CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                           d_keys_in, d_keys_out, size,
                                           d_stream));
  CHECK_CUDA(cudaFreeAsync(d_temp_storage, d_stream));
}

}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_CUDA_UTILS_H_
