
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_COUNTER_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_COUNTER_H_
#include <cooperative_groups.h>

#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/stream.h"
#define WARP_SIZE 32
namespace grape_gpu {

namespace dev {

class Counter {
 public:
  uint32_t* m_counter;

  DEV_HOST explicit Counter(uint32_t* counter) : m_counter(counter) {}

  DEV_INLINE void Add() {
    auto g = cooperative_groups::coalesced_threads();

    if (g.thread_rank() == 0) {
      atomicAdd(m_counter, g.size());
    }
  }

  DEV_INLINE void Reset() { *m_counter = 0; }

  DEV_INLINE uint32_t count() const { return *m_counter; }

 private:
};

}  // namespace dev

static __global__ void ResetCounters(uint32_t* counters,
                                     uint32_t num_counters) {
  if (TID_1D < num_counters)
    counters[TID_1D] = 0;
}

class Counter {
  enum { WS = 32 };

  //
  // device buffer / counters
  //
  uint32_t* m_counters;
  uint32_t* m_host_counter;
  int32_t m_current_slot;

 public:
  Counter() : m_counters(nullptr), m_current_slot(-1) { Alloc(); }

  Counter(const Counter& other) = delete;
  Counter(Counter&& other) = delete;

  ~Counter() { Free(); }

  typedef dev::Counter DeviceObjectType;

 private:
  void Alloc() {
    CHECK_CUDA(cudaMalloc(&m_counters, WS * sizeof(uint32_t)));
    CHECK_CUDA(cudaMallocHost(&m_host_counter, sizeof(uint32_t)));
  }

  void Free() {
    CHECK_CUDA(cudaFree(m_counters));
    CHECK_CUDA(cudaFreeHost(m_host_counter));
  }

 public:
  DeviceObjectType DeviceObject() const {
    assert(m_current_slot >= 0 && m_current_slot < WS);
    return dev::Counter(m_counters + m_current_slot);
  }

  void ResetAsync(const Stream& stream) {
    m_current_slot = (m_current_slot + 1) % WS;
    if (m_current_slot == 0) {
      ResetCounters<<<1, WS, 0, stream.cuda_stream()>>>(m_counters, WS);
    }
  }

  uint32_t GetCount(const Stream& stream) const {
    assert(m_current_slot >= 0 && m_current_slot < WS);

    CHECK_CUDA(cudaMemcpyAsync(m_host_counter, m_counters + m_current_slot,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               stream.cuda_stream()));
    stream.Sync();

    return *m_host_counter;
  }
};

}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_COUNTER_H_
