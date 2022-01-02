
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_ATOMIC_SUM_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_ATOMIC_SUM_H_
#include "grape_gpu/utils/ipc_array.h"
#include "grape_gpu/utils/multi_device_barrier.h"

namespace grape_gpu {

namespace dev {
template <typename T>
class AtomicSum {
 public:
  explicit AtomicSum(ArrayView<T*> data) : data_(data) {}

  __device__ __forceinline__ T sum(const T& val, volatile T* ddd,
                                   PeerGroup& pg) {
    *data_[pg.device_rank()] = val;
    pg.sync();
    if (pg.thread_rank() == 0) {
      T sum = 0;

      for (int i = 0; i < pg.num_devices(); i++) {
        sum += *data_[i];
      }

      for (int i = 0; i < pg.num_devices(); i++) {
        *data_[i] = *ddd;
      }
      *ddd += sum;
    }
    pg.sync();
    return *data_[pg.device_rank()];
  }

 private:
  ArrayView<T*> data_;
};
}  // namespace dev

template <typename T>
class AtomicSum {
 public:
  AtomicSum() = default;

  AtomicSum(const grape::CommSpec& comm_spec, int device_rank,
            int num_gpus_required)
      : buffer_(comm_spec) {
    std::vector<int> dev_ranks(comm_spec.local_num());

    CHECK_LT(device_rank, num_gpus_required);
    MPI_Allgather(&device_rank, 1, MPI_INT, dev_ranks.data(), 1, MPI_INT,
                  comm_spec.local_comm());

    buffer_.Init(device_rank != -1 ? 1 : 0);

    for (int i = 0; i < comm_spec.local_num(); i++) {
      if (dev_ranks[i] != -1) {
        pointers_.push_back(buffer_.view(i).data());
      }
    }
  }

  dev::AtomicSum<T> DeviceObject() {
    return dev::AtomicSum<T>(ArrayView<T*>(pointers_));
  }

 private:
  IPCArray<T, IPCMemoryPlacement::kDevice> buffer_;
  thrust::device_vector<T*> pointers_;
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_ATOMIC_SUM_H_
