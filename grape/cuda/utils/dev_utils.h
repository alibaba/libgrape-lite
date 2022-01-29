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

#ifndef GRAPE_CUDA_UTILS_DEV_UTILS_H_
#define GRAPE_CUDA_UTILS_DEV_UTILS_H_
#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/launcher.h"

__device__ static const char logtable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1,    0,     1,     1,     2,     2,     2,     2,     3,     3,     3,
    3,     3,     3,     3,     3,     LT(4), LT(5), LT(5), LT(6), LT(6), LT(6),
    LT(6), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)};

namespace grape {
namespace cuda {

namespace dev {
DEV_HOST_INLINE size_t round_up(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

// Refer:https://github.com/gunrock/gunrock/blob/a7fc6948f397912ca0c8f1a8ccf27d1e9677f98f/gunrock/oprtr/intersection/cta.cuh#L84
DEV_INLINE unsigned ilog2(unsigned int v) {
  register unsigned int t, tt;
  if (tt = v >> 16)
    return ((t = tt >> 8) ? 24 + logtable[t] : 16 + logtable[tt]);
  else
    return ((t = v >> 8) ? 8 + logtable[t] : logtable[v]);
}
// Refer:
// https://forums.developer.nvidia.com/t/why-doesnt-runtime-library-provide-atomicmax-nor-atomicmin-for-float/164171/7
DEV_INLINE float atomicMinFloat(float* addr, float value) {
  if (isnan(value)) {
    return *addr;
  }
  value += 0.0f;
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMin(reinterpret_cast<int*>(addr),
                                       __float_as_int(value)))
            : __uint_as_float(atomicMax(reinterpret_cast<unsigned int*>(addr),
                                        __float_as_uint(value)));

  return old;
}

template <typename T>
DEV_INLINE bool BinarySearch(const ArrayView<T>& array, const T& target) {
  size_t l = 0;
  size_t r = array.size();
  while (l < r) {
    size_t m = (l + r) >> 1;
    const T& elem = array[m];

    if (elem == target) {
      return true;
    } else if (elem > target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return false;
}

template <typename T>
DEV_INLINE bool BinarySearchWarp(const ArrayView<T>& array, const T& target) {
  size_t cosize = 32;
  size_t lane = threadIdx.x & (cosize - 1);
  size_t worknum = array.size();
  size_t x = worknum / cosize;
  size_t y = worknum % cosize;
  // size_t per_work = worknum / cosize + (lane < worknum % cosize);
  size_t l = lane * x + (lane < y ? lane : y);
  size_t r = l + x + (lane < y ? 1 : 0);
  bool found = false;
  __syncwarp();
  while (l < r) {
    size_t m = (l + r) >> 1;
    const T& elem = array[m];

    if (elem == target) {
      found = true;
      break;
    } else if (elem > target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  __syncwarp();
  return __any_sync(__activemask(), found);
}

template <int NT, typename T>
DEV_INLINE int BinarySearch(const T* arr, const T& key) {
  int mid = ((NT >> 1) - 1);

  if (NT > 512)
    mid = arr[mid] > key ? mid - 256 : mid + 256;
  if (NT > 256)
    mid = arr[mid] > key ? mid - 128 : mid + 128;
  if (NT > 128)
    mid = arr[mid] > key ? mid - 64 : mid + 64;
  if (NT > 64)
    mid = arr[mid] > key ? mid - 32 : mid + 32;
  if (NT > 32)
    mid = arr[mid] > key ? mid - 16 : mid + 16;
  mid = arr[mid] > key ? mid - 8 : mid + 8;
  mid = arr[mid] > key ? mid - 4 : mid + 4;
  mid = arr[mid] > key ? mid - 2 : mid + 2;
  mid = arr[mid] > key ? mid - 1 : mid + 1;
  mid = arr[mid] > key ? mid : mid + 1;

  return mid;
}

void WarmupNccl(const grape::CommSpec& comm_spec, const Stream& stream,
                std::shared_ptr<ncclComm_t>& nccl_comm) {
  auto fnum = comm_spec.fnum();
  auto fid = comm_spec.fid();
  std::vector<thrust::device_vector<char>> buf_in(fnum), buf_out(fnum);

  for (int _ = 0; _ < 10; _++) {
    size_t length = 4 * 1024 * 1024;

    CHECK_NCCL(ncclGroupStart());

    for (fid_t i = 1; i < fnum; ++i) {
      fid_t src_fid = (fid + i) % fnum;
      int peer = comm_spec.FragToWorker(src_fid);

      buf_in[src_fid].resize(length);
      CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(buf_in[src_fid].data()),
                          length, ncclChar, peer, *nccl_comm,
                          stream.cuda_stream()));
    }

    for (fid_t i = 1; i < fnum; ++i) {
      fid_t dst_fid = (fid + fnum - i) % fnum;
      int peer = comm_spec.FragToWorker(dst_fid);

      buf_out[dst_fid].resize(length);
      CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(buf_out[dst_fid].data()),
                          length, ncclChar, peer, *nccl_comm,
                          stream.cuda_stream()));
    }

    CHECK_NCCL(ncclGroupEnd());
    stream.Sync();
  }
}

template <typename T>
void ncclSendRecv(const grape::CommSpec& comm_spec, const Stream& stream,
                  std::shared_ptr<ncclComm_t>& nccl_comm,
                  const std::vector<int>& migrate_to,
                  const ArrayView<T>& send_buf, ArrayView<T> recv_buf) {
  int to_rank = migrate_to[comm_spec.worker_id()];

  if (to_rank != -1) {
    size_t send_size = send_buf.size();
    MPI_Send(&send_size, 1, MPI_UINT64_T, to_rank, 1, comm_spec.comm());
    CHECK_NCCL(ncclSend(send_buf.data(), sizeof(T) * send_buf.size(), ncclChar,
                        to_rank, *nccl_comm, stream.cuda_stream()));
  } else {
    for (int src_worker_id = 0; src_worker_id < comm_spec.worker_num();
         src_worker_id++) {
      if (migrate_to[src_worker_id] == comm_spec.worker_id()) {
        size_t recv_size;
        MPI_Status stat;
        MPI_Recv(&recv_size, 1, MPI_UINT64_T, src_worker_id, 1,
                 comm_spec.comm(), &stat);
        CHECK_NCCL(ncclRecv(recv_buf.data(), sizeof(T) * recv_size, ncclChar,
                            src_worker_id, *nccl_comm, stream.cuda_stream()));
      }
    }
  }
}

}  // namespace dev
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_DEV_UTILS_H_
