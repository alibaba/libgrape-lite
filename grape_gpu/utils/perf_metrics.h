#ifndef GRAPEGPU_GRAPE_GPU_UTILS_PERF_METRICS_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_PERF_METRICS_H_
#include "grape/worker/comm_spec.h"
#include "grape_gpu/utils/ipc_array.h"

namespace grape_gpu {
__global__ void delay(volatile int* flag,
                      unsigned long long timeout_clocks = 10000000) {
  // Wait until the application notifies us that it has completed queuing up the
  // experiment, or timeout and exit, allowing the application to make progress
  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (!*flag) {
    sample_clock = clock64();

    if (sample_clock - start_clock > timeout_clocks) {
      break;
    }
  }
}

struct Metrics {
  double bi_bandwidth_gb{};     // bidirectional
  double read_bandwidth_gb{};   // unidirectional, read from remote and write
                                // to local
  double write_bandwidth_gb{};  // read from local and write to remote
  bool p2p{};
  bool atomic{};
};

class PerfMetrics {
  using test_unit_t = int4;
  static constexpr int metric_size_mb = 32;

 public:
  explicit PerfMetrics(const grape::CommSpec& comm_spec)
      : comm_spec_(comm_spec), ipc_array_(comm_spec) {
    comm_spec_.Dup();
    auto n_elem = metric_size_mb * 1024 * 1024 / sizeof(test_unit_t);
    int device;

    CHECK_CUDA(cudaGetDevice(&device));
    communicator_.InitCommunicator(comm_spec_.local_comm());
    communicator_.AllGather(device, devices_);

    ipc_array_.Init(n_elem);
    d2d_buffer_.resize(n_elem);
  }

  void Evaluate() {
    int repeat = 5;
    auto local_num = comm_spec_.local_num();
    std::vector<Metrics> metrics(local_num);
    Stream stream;

    for (int local_id = 0; local_id < local_num; local_id++) {
      if (local_id == comm_spec_.local_id()) {
        for (int peer_id = 0; peer_id < local_num; peer_id++) {
          metrics[peer_id].read_bandwidth_gb =
              evalBandwidth(stream, peer_id, local_id, false, repeat);
          metrics[peer_id].write_bandwidth_gb =
              evalBandwidth(stream, local_id, peer_id, false, repeat);
          metrics[peer_id].bi_bandwidth_gb =
              evalBandwidth(stream, local_id, peer_id, true, repeat);
          if (devices_[local_id] != devices_[peer_id]) {
            int val;
            CHECK_CUDA(cudaDeviceGetP2PAttribute(
                &val, cudaDevP2PAttrAccessSupported, devices_[local_id],
                devices_[peer_id]));
            metrics[peer_id].p2p = val == 1;
            CHECK_CUDA(cudaDeviceGetP2PAttribute(
                &val, cudaDevP2PAttrNativeAtomicSupported, devices_[local_id],
                devices_[peer_id]));
            metrics[peer_id].atomic = val == 1;
          }
        }
      }
      MPI_Barrier(comm_spec_.local_comm());
    }

    communicator_.AllGather(metrics, metrics_);
  }

  Metrics metrics(int src_local_id, int dst_local_id) const {
    return metrics_[src_local_id][dst_local_id];
  }

 private:
  double evalBandwidth(Stream& stream, int src_local_id, int dst_local_id,
                       bool bidirectional, int repeat) {
    auto src_view = ipc_array_.view(src_local_id);
    auto dst_view = ipc_array_.view(dst_local_id);
    cudaEvent_t start, stop;
    volatile int* flag;
    size_t size_byte = src_view.size() * sizeof(test_unit_t);

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(
        cudaHostAlloc((void**) &flag, sizeof(*flag), cudaHostAllocPortable));

    *flag = 0;
    delay<<<1, 1, 0, stream.cuda_stream()>>>(flag);

    CHECK_CUDA(cudaEventRecord(start, stream.cuda_stream()));

    for (int r = 0; r < repeat; r++) {
      if (src_local_id == dst_local_id) {
        CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d2d_buffer_.data()),
                                   src_view.data(), size_byte,
                                   cudaMemcpyDefault, stream.cuda_stream()));
        if (bidirectional) {
          CHECK_CUDA(cudaMemcpyAsync(
              dst_view.data(), thrust::raw_pointer_cast(d2d_buffer_.data()),
              size_byte, cudaMemcpyDefault, stream.cuda_stream()));
        }
      } else {
        CHECK_CUDA(cudaMemcpyAsync(dst_view.data(), src_view.data(), size_byte,
                                   cudaMemcpyDefault, stream.cuda_stream()));
        if (bidirectional) {
          CHECK_CUDA(cudaMemcpyAsync(src_view.data(), dst_view.data(),
                                     size_byte, cudaMemcpyDefault,
                                     stream.cuda_stream()));
        }
      }
    }
    CHECK_CUDA(cudaEventRecord(stop, stream.cuda_stream()));

    *flag = 1;
    stream.Sync();
    CHECK_CUDA(cudaFreeHost((void*) flag));

    float time;
    cudaEventElapsedTime(&time, start, stop);

    size_byte *= repeat;
    if (bidirectional) {
      size_byte *= 2;
    }
    // same device
    if (src_local_id == dst_local_id) {
      size_byte *= 2;
    }

    return ((double) size_byte / 1024 / 1024 / 1024) / (time / 1000);
  }

  grape::CommSpec comm_spec_;
  grape::Communicator communicator_;
  std::vector<int> devices_;
  IPCArray<test_unit_t, IPCMemoryPlacement::kDevice> ipc_array_;
  thrust::device_vector<test_unit_t> d2d_buffer_;
  std::vector<std::vector<Metrics>> metrics_;
};

}  // namespace grape_gpu

namespace grape {
inline grape::OutArchive& operator>>(grape::OutArchive& archive,
                                     grape_gpu::Metrics& metrics) {
  archive >> metrics.read_bandwidth_gb >> metrics.write_bandwidth_gb >>
      metrics.bi_bandwidth_gb >> metrics.p2p >> metrics.atomic;
  return archive;
}

inline grape::InArchive& operator<<(grape::InArchive& archive,
                                    const grape_gpu::Metrics& metrics) {
  archive << metrics.read_bandwidth_gb << metrics.write_bandwidth_gb
          << metrics.bi_bandwidth_gb << metrics.p2p << metrics.atomic;
  return archive;
}
}  // namespace grape
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_PERF_METRICS_H_
