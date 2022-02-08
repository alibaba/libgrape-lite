#ifndef common_H
#define common_H
#define LONG_PTR
#include <stdint.h>

namespace CUDASTL {
// =====================================
// tools
// align n to b bytes
template <class T>
__device__ T Align(T n, uint32_t b) {
  return ((uint32_t) n & (b - 1)) == NULL ? n
                                          : n + b - ((uint32_t) n & (b - 1));
}

template <class T1, class T2, class T3>
__device__ bool CAS64(T1* addr, T2 old_val, T3 new_val) {
  return *(unsigned long long*) (&old_val) ==
         atomicCAS((unsigned long long*) addr,
                   *(unsigned long long*) (&old_val),
                   *(unsigned long long*) (&new_val));
}

template <class T1, class T2, class T3>
__device__ bool CAS32(T1* addr, T2 old_val, T3 new_val) {
  return *(uint32_t*) (&old_val) == atomicCAS((uint32_t*) addr,
                                              *(uint32_t*) (&old_val),
                                              *(uint32_t*) (&new_val));
}

template <class T1, class T2, class T3>
__device__ bool CASPTR(T1* addr, T2 old_val, T3 new_val) {
#ifdef LONG_PTR
  return CAS64(addr, old_val, new_val);
#else
  return CAS32(addr, old_val, new_val);
#endif
}

template <class T1, class T2>
__device__ uint32_t ADD32(T1* addr, T2 val) {
  return atomicAdd((uint32_t*) addr, *(uint32_t*) (&val));
}

__device__ uint32_t get_thread_id() {
  uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t blockSize = blockDim.z * blockDim.y * blockDim.x;
  uint32_t thread_id =
      threadIdx.z * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  return block_id * blockSize + thread_id;
}

template <class T>
__host__ T* CreateDeviceVar(T& h_var) {
  T* d_addr;
  cudaMalloc((void**) &d_addr, sizeof(T));
  cudaMemcpy(d_addr, &h_var, sizeof(T), cudaMemcpyHostToDevice);
  return d_addr;
}

template <class T>
__host__ T DownloadDeviceVar(T* d_ptr) {
  T temp;
  cudaMemcpy(&temp, d_ptr, sizeof(T), cudaMemcpyDeviceToHost);
  return temp;
}
};  // namespace CUDASTL

#endif
