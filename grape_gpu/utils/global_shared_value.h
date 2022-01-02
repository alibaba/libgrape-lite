
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_GLOBAL_SHARED_VALUE_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_GLOBAL_SHARED_VALUE_H_
#include "grape_gpu/utils/ipc_array.h"
namespace grape_gpu {

template <typename T>
class GlobalSharedValue {
 public:
  GlobalSharedValue() = default;

  GlobalSharedValue(const grape::CommSpec& comm_spec, int allocate_on)
      : allocate_on_(allocate_on), d_buffer_(comm_spec) {
    d_buffer_.Init(comm_spec.local_id() == allocate_on ? 1 : 0);
    h_buffer_.resize(1);
  }

  void set(const T& t, const Stream& stream) {
    h_buffer_[0] = t;
    CHECK_CUDA(cudaMemcpyAsync(
        data(), thrust::raw_pointer_cast(h_buffer_.data()), sizeof(T),
        cudaMemcpyHostToDevice, stream.cuda_stream()));
  }

  void set(const T& t) {
    CHECK_CUDA(cudaMemcpy(data(), &t, sizeof(T), cudaMemcpyHostToDevice));
  }

  T get(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(h_buffer_.data()),
                               data(), sizeof(T), cudaMemcpyDeviceToHost,
                               stream.cuda_stream()));
    stream.Sync();
    return h_buffer_[0];
  }

  T* data() { return d_buffer_.view(allocate_on_).data(); }

 private:
  int allocate_on_;
  IPCArray<T, IPCMemoryPlacement::kDevice> d_buffer_;
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>
      h_buffer_;
};

}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_GLOBAL_SHARED_VALUE_H_
