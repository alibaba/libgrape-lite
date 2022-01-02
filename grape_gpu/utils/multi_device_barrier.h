
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_MULTI_DEVICE_BARRIER_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_MULTI_DEVICE_BARRIER_H_
#include <cooperative_groups.h>

#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/ipc_array.h"

namespace grape_gpu {

struct MultiDeviceData {
  unsigned char* hostMemoryArrivedList{};
  unsigned int numDevices{};
  unsigned int deviceRank{};
};

// Class used for coordination of multiple devices.
class PeerGroup {
  const MultiDeviceData& data;
  const cooperative_groups::grid_group& grid;

  __device__ unsigned char load_arrived(unsigned char* arrived) const {
#if __CUDA_ARCH__ < 700
    return *(volatile unsigned char*) arrived;
#else
    unsigned int result;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];"
                 : "=r"(result)
                 : "l"(arrived)
                 : "memory");
    return result;
#endif
  }

  __device__ void store_arrived(unsigned char* arrived,
                                unsigned char val) const {
#if __CUDA_ARCH__ < 700
    *(volatile unsigned char*) arrived = val;
#else
    unsigned int reg_val = val;
    asm volatile(
        "st.release.sys.global.u8 [%1], %0;" ::"r"(reg_val) "l"(arrived)
        : "memory");

    // Avoids compiler warnings from unused variable val.
    (void) (reg_val = reg_val);
#endif
  }

 public:
  __device__ PeerGroup(const MultiDeviceData& data,
                       const cooperative_groups::grid_group& grid)
      : data(data), grid(grid){};

  __device__ unsigned int size() const { return data.numDevices * grid.size(); }

  __device__ unsigned int thread_rank() const {
    return data.deviceRank * grid.size() + grid.thread_rank();
  }

  __device__ unsigned int num_devices() const {
    return data.numDevices;
  }

  __device__ unsigned int device_rank() const { return data.deviceRank; }

  __device__ void sync() const {
    grid.sync();

    // One thread from each grid participates in the sync.
    if (grid.thread_rank() == 0) {
      if (data.deviceRank == 0) {
        // Leader grid waits for others to join and then releases them.
        // Other GPUs can arrive in any order, so the leader have to wait for
        // all others.
        for (int i = 0; i < data.numDevices - 1; i++) {
          while (load_arrived(&data.hostMemoryArrivedList[i]) == 0)
            ;
        }
        for (int i = 0; i < data.numDevices - 1; i++) {
          store_arrived(&data.hostMemoryArrivedList[i], 0);
        }
        __threadfence_system();
      } else {
        // Other grids note their arrival and wait to be released.
        store_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1], 1);
        while (load_arrived(&data.hostMemoryArrivedList[data.deviceRank - 1]) ==
               1)
          ;
      }
    }

    grid.sync();
  }
};

class MultiDeviceBarrier {
 public:
  MultiDeviceBarrier() = default;

  MultiDeviceBarrier(const grape::CommSpec& comm_spec, int device_rank,
                     int num_gpus_required)
      : buffer_(comm_spec) {
    auto size = std::max(1ul, (num_gpus_required - 1) * sizeof(unsigned char));
    std::vector<int> dev_ranks(comm_spec.local_num());

    CHECK_LT(device_rank, num_gpus_required);
    MPI_Allgather(&device_rank, 1, MPI_INT, dev_ranks.data(), 1, MPI_INT,
                  comm_spec.local_comm());
    // find a node whose device_rank is zero
    for (int i = 0; i < comm_spec.local_num(); i++) {
      if (dev_ranks[i] == 0) {
        CHECK_EQ(leader_local_id_, -1);
        leader_local_id_ = i;
      }
    }
    CHECK_NE(leader_local_id_, -1);

    buffer_.Init(device_rank == 0 ? size : 0, 0);

    multi_device_data_.hostMemoryArrivedList =
        buffer_.view(leader_local_id_).data();
    multi_device_data_.numDevices = num_gpus_required;
    multi_device_data_.deviceRank = device_rank;
  }

  MultiDeviceData GetContext() { return multi_device_data_; }

  int leader_local_id() const { return leader_local_id_; }

 private:
  grape_gpu::IPCArray<unsigned char, IPCMemoryPlacement::kDevice> buffer_;
  MultiDeviceData multi_device_data_;
  int leader_local_id_{-1};
};

}  // namespace grape_gpu

#endif  // GRAPEGPU_GRAPE_GPU_UTILS_MULTI_DEVICE_BARRIER_H_
