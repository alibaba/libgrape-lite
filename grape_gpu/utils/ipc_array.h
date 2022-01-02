
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_IPC_ARRAY_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_IPC_ARRAY_H_
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "grape/worker/comm_spec.h"
#include "grape_gpu/fragment/id_parser.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/stream.h"

#if SIZE_MAX == UCHAR_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

namespace grape_gpu {
enum class IPCMemoryPlacement { kHost, kHostPinned, kDevice };

namespace dev {
template <typename T, typename VID_T>
class IPCArray {
 public:
  IPCArray() = default;
  DEV_HOST IPCArray(ArrayView<T>* data_ptrs, size_t* prefix, size_t fnum)
      : dev_ptrs_(data_ptrs), prefix_(prefix), fnum_(fnum) {
    id_parser_(fnum);
  }

  ArrayView<T> Local() { return dev_ptrs_[fid_]; }
  ArrayView<T> Remote(size_t fid) { return dev_ptrs_[fid]; }

 protected:
  IdParser<VID_T> id_parser_;
  ArrayView<T>* dev_ptrs_;
  size_t* prefix_;
  size_t fnum_;
  size_t fid_;
};
}  // namespace dev

template <typename T, IPCMemoryPlacement placement>
class IPCArray {
 public:
  IPCArray() : local_id_(-1), local_num_(0), win_(MPI_WIN_NULL) {}

  explicit IPCArray(const grape::CommSpec& comm_spec)
      : comm_spec_(std::make_shared<grape::CommSpec>(comm_spec)),
        local_id_(comm_spec.local_id()),
        local_num_(comm_spec.local_num()),
        device_data_(comm_spec_->local_num()),
        h_mirror_views_(comm_spec_->local_num()),
        d_mirror_views_(comm_spec_->local_num()),
        win_(MPI_WIN_NULL),
        h_views_(local_num_),
        sizes_(local_num_, 0) {
    comm_spec_->Dup();
  }

  ~IPCArray() { free(); }

  IPCArray(const IPCArray& rhs) = delete;

  IPCArray& operator=(const IPCArray& rhs) = delete;

  IPCArray(IPCArray&& rhs) noexcept
      : local_id_(-1), local_num_(0), win_(MPI_WIN_NULL) {
    *this = std::move(rhs);
  }

  IPCArray& operator=(IPCArray&& rhs) noexcept {
    if (this != &rhs) {
      comm_spec_ = std::move(rhs.comm_spec_);
      local_id_ = rhs.local_id_;
      local_num_ = rhs.local_num_;
      device_data_ = std::move(rhs.device_data_);
      win_ = rhs.win_;
      prefix_ = std::move(rhs.prefix_);
      h_views_ = std::move(rhs.h_views_);
      d_views_ = std::move(rhs.d_views_);
      sizes_ = std::move(rhs.sizes_);

      rhs.local_id_ = -1;
      rhs.local_num_ = 0;
      rhs.win_ = MPI_WIN_NULL;
    }
    return *this;
  }

  void Init(size_t size) {
    if (placement == IPCMemoryPlacement::kDevice) {
      device_data_[local_id_].resize(size);
      initForDevice(ArrayView<T>(device_data_[local_id_]));
    } else {
      initForHost(size, placement == IPCMemoryPlacement::kHostPinned);
    }
    MPI_Barrier(comm_spec_->local_comm());
  }

  void InitWithDevicePtr(T* ptr, size_t size) {
    static_assert(placement == IPCMemoryPlacement::kDevice, "Invalid usage");
    initForDevice(ArrayView<T>(ptr, size));
    MPI_Barrier(comm_spec_->local_comm());
  }

  void Init(size_t size, const T& value) {
    Init(size);
    auto view = h_views_[local_id_];
    if (placement == IPCMemoryPlacement::kDevice) {
      thrust::fill_n(thrust::device, device_data_[local_id_].data(), size,
                     value);
    } else {
      std::fill_n(view.data(), view.size(), value);
    }
    MPI_Barrier(comm_spec_->local_comm());
  }

  void Clear() {
    auto view = h_views_[local_id_];
    if (placement == IPCMemoryPlacement::kDevice) {
      thrust::fill_n(thrust::device, device_data_[local_id_].data(),
                     device_data_[local_id_].size(), 0);
    } else {
      std::fill_n(view.data(), view.size(), 0);
    }
    MPI_Barrier(comm_spec_->local_comm());
  }

  ArrayView<T> local_view() { return view(local_id_); }

  ArrayView<T> view(int local_id) { return h_views_[local_id]; }

  ArrayView<T> mirror_view(int local_id) { return h_mirror_views_[local_id]; }

  ArrayView<ArrayView<T>> device_views() {
    return ArrayView<ArrayView<T>>(d_views_);
  }

  ArrayView<ArrayView<T>> host_views() {
    return ArrayView<ArrayView<T>>(h_views_.data(), h_views_.size());
  }

  size_t size(int local_id) const { return sizes_[local_id]; }

  int array_count() const { return local_num_; }

  void SyncToMirror() {
    for (int local_id = 0; local_id < local_num_; local_id++) {
      auto view = h_views_[local_id];
      auto& mirror = device_data_[local_id];

      if (local_id != local_id_) {
        mirror.resize(view.size());
        thrust::copy(thrust::device, view.begin(), view.end(), mirror.begin());
      }
      h_mirror_views_[local_id] = ArrayView<T>(mirror);
    }
    d_mirror_views_ = h_mirror_views_;
  }

  void Swap(IPCArray& rhs) {
    std::swap(comm_spec_, rhs.comm_spec_);
    std::swap(local_id_, rhs.local_id_);
    std::swap(local_num_, rhs.local_num_);
    std::swap(device_data_, rhs.device_data_);
    std::swap(win_, rhs.win_);
    prefix_.swap(rhs.prefix_);
    h_views_.swap(rhs.h_views_);
    d_views_.swap(rhs.d_views_);
    h_mirror_views_.swap(rhs.h_mirror_views_);
    d_mirror_views_.swap(rhs.d_mirror_views_);
    sizes_.swap(rhs.sizes_);
  }

  template <typename VID_T>
  dev::IPCArray<T, VID_T> DeviceObject() {
    return dev::IPCArray<T, VID_T>(thrust::raw_pointer_cast(d_views_.data()),
                                   thrust::raw_pointer_cast(prefix_.data()),
                                   local_num_, local_id_);
  }

  int local_id() const { return local_id_; }

  int local_num() const { return local_num_; }

 protected:
  void free() {
    if (placement == IPCMemoryPlacement::kDevice) {
      for (int local_id = 0; local_id < local_num_; local_id++) {
        if (local_id != local_id_ && h_views_[local_id].data() != nullptr) {
          CHECK_CUDA(cudaIpcCloseMemHandle(h_views_[local_id].data()));
        }
      }
    } else {
      if (win_ != MPI_WIN_NULL) {
        MPI_Win_fence(0, win_);
        MPI_Win_free(&win_);
      }
    }
    // Object is moved
    if (comm_spec_ != nullptr) {
      MPI_Barrier(comm_spec_->local_comm());
    }
  }

  void initForDevice(ArrayView<T> view) {
    cudaIpcMemHandle_t mem_handle;
    size_t size = view.size();
    std::vector<cudaIpcMemHandle_t> mem_handles(local_num_);

    initSizes(size);

    if (size > 0) {
      CHECK_CUDA(cudaIpcGetMemHandle(&mem_handle, view.data()));
    }
    MPI_Allgather(&mem_handle, sizeof(cudaIpcMemHandle_t), MPI_CHAR,
                  mem_handles.data(), sizeof(cudaIpcMemHandle_t), MPI_CHAR,
                  comm_spec_->local_comm());

    // open mem_handle
    for (int local_id = 0; local_id < local_num_; local_id++) {
      T* ptr = nullptr;
      if (local_id != local_id_) {
        if (sizes_[local_id] > 0) {
          CHECK_CUDA(cudaIpcOpenMemHandle((void**) &ptr, mem_handles[local_id],
                                          cudaIpcMemLazyEnablePeerAccess));
        }
      } else {
        ptr = view.data();
      }
      h_views_[local_id] = ArrayView<T>(ptr, sizes_[local_id]);
    }
    d_views_ = h_views_;
  }

  void initForHost(size_t size, bool pinned) {
    initSizes(size);

    size_t total_size = prefix_[local_num_];
    T* buffer = nullptr;

    if (comm_spec_->local_id() == 0) {
      MPI_Win_allocate_shared(total_size * sizeof(T), sizeof(T), MPI_INFO_NULL,
                              comm_spec_->local_comm(), &buffer, &win_);
    } else {
      MPI_Aint size_in_bytes;
      int unit_size_in_bytes;
      MPI_Win_allocate_shared(total_size * sizeof(T), sizeof(T), MPI_INFO_NULL,
                              comm_spec_->local_comm(), &buffer, &win_);
      MPI_Win_shared_query(win_, 0, &size_in_bytes, &unit_size_in_bytes,
                           &buffer);
    }

    if (pinned) {
      CHECK_CUDA(cudaHostRegister(buffer, total_size * sizeof(T),
                                  cudaHostRegisterDefault));
    }

    MPI_Win_fence(0, win_);
    for (int local_id = 0; local_id < local_num_; local_id++) {
      h_views_[local_id] =
          ArrayView<T>(buffer + prefix_[local_id], sizes_[local_id]);
    }
    d_views_ = h_views_;
  }

  void initSizes(size_t size) {
    prefix_.resize(local_num_ + 1, 0);

    MPI_Allgather(&size, 1, my_MPI_SIZE_T, sizes_.data(), 1, my_MPI_SIZE_T,
                  comm_spec_->local_comm());

    for (int i = 0; i < local_num_; ++i) {
      prefix_[i + 1] = prefix_[i] + sizes_[i];
    }
  }

  std::shared_ptr<grape::CommSpec> comm_spec_;
  int local_id_, local_num_;

  std::vector<thrust::device_vector<T>> device_data_;
  MPI_Win win_;

  thrust::device_vector<size_t> prefix_;
  thrust::host_vector<ArrayView<T>> h_views_;
  thrust::device_vector<ArrayView<T>> d_views_;
  thrust::host_vector<ArrayView<T>> h_mirror_views_;
  thrust::device_vector<ArrayView<T>> d_mirror_views_;

  std::vector<size_t> sizes_;
};

}  // namespace grape_gpu

#endif  // GRAPEGPU_GRAPE_GPU_UTILS_IPC_ARRAY_H_
