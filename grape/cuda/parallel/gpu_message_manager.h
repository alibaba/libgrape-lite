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

#ifndef GRAPE_CUDA_PARALLEL_GPU_MESSAGE_MANAGER_H_
#define GRAPE_CUDA_PARALLEL_GPU_MESSAGE_MANAGER_H_

#include <mpi.h>
#include <thrust/pair.h>

#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grape/cuda/parallel/message_kernels.h"
#include "grape/cuda/serialization/in_archive.h"
#include "grape/cuda/serialization/out_archive.h"
#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/event.h"
#include "grape/cuda/utils/markers.h"
#include "grape/cuda/utils/stream.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/util.h"

namespace grape {
namespace cuda {
namespace dev {
class MessageManager {
 public:
  explicit MessageManager(const ArrayView<InArchive>& to_send)
      : to_send_(to_send) {}

  // We can not use warp aggregation for here, because the different InArchive
  // objects will be used
  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                         const typename GRAPH_T::vertex_t& v,
                                         const MESSAGE_T& msg) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytes(thrust::make_pair(frag.GetOuterVertexGid(v), msg));
  }

  template <typename GRAPH_T>
  DEV_INLINE void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                         const typename GRAPH_T::vertex_t& v) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytes(frag.GetOuterVertexGid(v));
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SyncStateOnOuterVertexWarpOpt(
      const GRAPH_T& frag, const typename GRAPH_T::vertex_t& v,
      const MESSAGE_T& msg) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytesWarpOpt(
        fid, thrust::make_pair(frag.GetOuterVertexGid(v), msg));
  }

  template <typename GRAPH_T>
  DEV_INLINE void SyncStateOnOuterVertexWarpOpt(
      const GRAPH_T& frag, const typename GRAPH_T::vertex_t& v) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytesWarpOpt(fid, frag.GetOuterVertexGid(v));
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SendMsgThroughIEdges(const GRAPH_T& frag,
                                       const typename GRAPH_T::vertex_t& v,
                                       const MESSAGE_T& msg) {
    DestList dsts = frag.IEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);

    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid].AddBytes(thrust::make_pair(gid, msg));
    }
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SendMsgThroughOEdges(const GRAPH_T& frag,
                                       const typename GRAPH_T::vertex_t& v,
                                       const MESSAGE_T& msg) {
    DestList dsts = frag.OEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);

    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid].AddBytes(thrust::make_pair(gid, msg));
    }
  }

  template <typename GRAPH_T>
  DEV_INLINE void SendMsgThroughOEdges(const GRAPH_T& frag,
                                       const typename GRAPH_T::vertex_t& v) {
    DestList dsts = frag.OEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);

    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid].AddBytes(gid);
    }
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SendMsgThroughEdges(const GRAPH_T& frag,
                                      const typename GRAPH_T::vertex_t& v,
                                      const MESSAGE_T& msg) {
    DestList dsts = frag.IOEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);

    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid].AddBytes(thrust::make_pair(gid, msg));
    }
  }

  template <typename MESSAGE_T>
  DEV_INLINE void SendToFragment(fid_t dst_fid, const MESSAGE_T& msg) {
    to_send_[dst_fid].AddBytes(msg);
  }

  template <typename MESSAGE_T>
  DEV_INLINE void SendToFragmentWarpOpt(fid_t dst_fid, const MESSAGE_T& msg) {
    to_send_[dst_fid].AddBytesWarp(msg);
  }

 private:
  ArrayView<InArchive> to_send_;
};
}  // namespace dev

class GPUMessageManager {
 public:
  /**
   * @brief Initialize message manager.
   *
   * @param comm MPI_Comm object.
   */
  void Init(const grape::CommSpec& comm_spec) {
    comm_spec_ = comm_spec;
    fnum_ = comm_spec.fnum();
    fid_ = comm_spec.fid();
    lengths_out_.resize(fnum_);
    lengths_in_.resize(fnum_ * fnum_);

    to_send_.Init(fnum_);
    d_to_send_.resize(fnum_);
    to_recv_.Init(fnum_);
    pinned_to_recv_.resize(fnum_);
    d_to_recv_.resize(fnum_);

    ncclUniqueId id;
    if (comm_spec.worker_id() == grape::kCoordinatorRank) {
      CHECK_NCCL(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, comm_spec.comm());

    int dev_id = comm_spec.local_id();
    nccl_comm_ =
        std::shared_ptr<ncclComm_t>(new ncclComm_t, [](ncclComm_t* comm) {
          CHECK_NCCL(ncclCommDestroy(*comm));
          delete comm;
        });

    CHECK_CUDA(cudaSetDevice(dev_id));
    CHECK_NCCL(ncclCommInitRank(nccl_comm_.get(), comm_spec.worker_num(), id,
                                comm_spec.worker_id()));

    // Warm-up
    dev::WarmupNccl(comm_spec, comm_stream_, nccl_comm_);
    DeviceWarmup(fnum_);
    comm_stream_.Sync();
  }

  void InitBuffer(size_t send_buffer_capacity, size_t recv_buffer_capacity) {
    for (fid_t fid = 0; fid < fnum_; fid++) {
      if (fid != fid_) {
        to_send_.resize(fid, send_buffer_capacity);
        d_to_send_[fid] = to_send_.DeviceObject(fid);
        to_recv_.resize(fid, recv_buffer_capacity);
      }
    }
  }

  void DropBuffer() {
    for (fid_t fid = 0; fid < fnum_; fid++) {
      if (fid != fid_) {
        to_send_.resize(fid, 0);
        d_to_send_[fid] = to_send_.DeviceObject(fid);
        to_recv_.resize(fid, 0);
      }
    }
  }

  /**
   * @brief This function will be called before Init step of applications.
   */
  void Start() {}

  /**
   * @brief This function will be called before each evaluation step of
   * applications.
   */
  void StartARound() {
    force_continue_ = false;
    sent_size_ = 0;

    computation_finished_ = Event::Create();
    to_send_.Clear(compute_stream_);
  }

  /**
   * @brief This function will be called after each evaluation step of
   * applications.
   */
  void FinishARound() {
    to_terminate_ = syncLengths();

    if (to_terminate_) {
      return;
    }

    to_recv_.Clear(comm_stream_);
    comm_stream_.Sync();

#ifdef PROFILING
    RangeMarker marker(true, "nccl calling");
#endif
    double memcpy_time = grape::GetCurrentTime();

    CHECK_NCCL(ncclGroupStart());

    for (fid_t i = 1; i < fnum_; ++i) {
      auto src_fid = (fid_ + i) % fnum_;
      auto peer = comm_spec_.FragToWorker(src_fid);
      auto length = lengths_in_[src_fid * fnum_ + fid_];

      to_recv_.resize(src_fid, length);
      if (length > 0) {
        CHECK_NCCL(ncclRecv(to_recv_.data(src_fid), length, ncclChar, peer,
                            *nccl_comm_, comm_stream_.cuda_stream()));
      }
      pinned_to_recv_[src_fid] = to_recv_.DeviceObject(src_fid);
    }

    for (fid_t i = 1; i < fnum_; ++i) {
      auto dst_fid = (fid_ + fnum_ - i) % fnum_;
      auto peer = comm_spec_.FragToWorker(dst_fid);
      auto* data = to_send_.data(dst_fid);
      auto length = lengths_out_[dst_fid];

      if (length > 0) {
        CHECK_NCCL(ncclSend(data, length, ncclChar, peer, *nccl_comm_,
                            comm_stream_.cuda_stream()));
      }
    }
    CHECK_NCCL(ncclGroupEnd());
#ifdef PROFILING
    marker.Stop();
#endif

    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_to_recv_.data()),
                               thrust::raw_pointer_cast(pinned_to_recv_.data()),
                               sizeof(dev::OutArchive) * fnum_,
                               cudaMemcpyHostToDevice,
                               comm_stream_.cuda_stream()));

    comm_stream_.Sync();

#ifdef PROFILING
    float size_in_mb = static_cast<float>(sent_size_) / 1024 / 1024;
    memcpy_time = grape::GetCurrentTime() - memcpy_time;

    total_memcpy_time_ += memcpy_time;

    VLOG(2) << "Worker " << fid_ << " Copy size " << size_in_mb << " MB, "
            << "Time: " << memcpy_time * 1000 << " ms, "
            << "Acc Time: " << total_memcpy_time_ * 1000 << " ms, "
            << "Bandwith: " << size_in_mb / memcpy_time << " MB/S";
#endif
  }

  /**
   * @brief This function will be called after the evaluation of applications.
   */
  void Finalize() const {
#ifdef PROFILING
    VLOG(2) << "Memory copy time: " << total_memcpy_time_ * 1000 << " ms";
#endif
  }

  /**
   * @brief This function will be called by worker after a step to determine
   * whether evaluation is terminated.
   *
   * @return Whether evaluation is terminated.
   */
  bool ToTerminate() const { return to_terminate_; }

  /**
   * @brief Get size of messages sent by this message manager instance.
   * The return value is valid only after FinishARound is called.
   * StartARound will reset the value to zero.
   *
   * @return Size of messages sent by this message manager instance.
   */
  size_t GetMsgSize() const { return sent_size_; }

  /**
   * @brief Force continue to evaluate one more round even if all workers stop
   * sending message.
   *
   * This function can be called by applications.
   */
  void ForceContinue() { force_continue_ = true; }

  double GetAccumulatedCommTime() const { return total_memcpy_time_; }

  dev::MessageManager DeviceObject() {
    return dev::MessageManager(ArrayView<dev::InArchive>(d_to_send_));
  }

  /**
   * @brief For some GPU servers, the first kernel always takes a long time.
   * This is a Dummy function to warm up the device.
   *
   * This function can be called by applications.
   */
  void DeviceWarmup(size_t np) {
    LaunchKernel(
        compute_stream_,
        [=] __device__(ArrayView<dev::OutArchive> recvs) {
          if (recvs.size() != np) {
            printf("Panic\n");
          }
        },
        ArrayView<dev::OutArchive>(d_to_recv_));
    compute_stream_.Sync();
  }

  template <typename GRAPH_T, typename MESSAGE_T = grape::EmptyType,
            typename FUNC_T>
  inline void ParallelProcess(const GRAPH_T& frag, FUNC_T func) {
    int grid_size = 256, block_size = 256;

#ifdef PROFILING
    RangeMarker marker(true, "ParallelProcess");
#endif

    dev::ProcessMsg<GRAPH_T, MESSAGE_T, FUNC_T>
        <<<grid_size, block_size, 0, compute_stream_.cuda_stream()>>>(
            ArrayView<dev::OutArchive>(d_to_recv_), frag, func);
    compute_stream_.Sync();
#ifdef PROFILING
    marker.Stop();
#endif
  }

  template <typename MESSAGE_T = grape::EmptyType, typename FUNC_T>
  inline void ParallelProcess(FUNC_T func) {
    int grid_size = 256, block_size = 256;

    // TODO(mengke): Fuse
    for (fid_t src_fid = 0; src_fid < fnum_; src_fid++) {
      if (src_fid != fid_) {
        dev::ProcessMsg<MESSAGE_T, FUNC_T>
            <<<grid_size, block_size, 0, compute_stream_.cuda_stream()>>>(
                to_recv_.DeviceObject(src_fid), func);
      }
    }
    compute_stream_.Sync();
  }

  Stream& stream() { return compute_stream_; }

  ncclComm_t nccl_comm() { return *nccl_comm_; }

 private:
  bool syncLengths() {
    computation_finished_.Record(compute_stream_);
    computation_finished_.Wait(comm_stream_);

    auto sent_sizes = to_send_.size(comm_stream_);

    for (fid_t fid = 0; fid < fnum_; fid++) {
      auto sent_size = sent_sizes[fid];

      sent_size_ += sent_size;
      lengths_out_[fid] = sent_size;
    }

    if (force_continue_ && lengths_out_[fid_] == 0) {
      lengths_out_[fid_]++;
    }

    {
#ifdef PROFILING
      RangeMarker marker(true, "MsgManager - Allgather");
#endif
      MPI_Allgather(&lengths_out_[0], fnum_ * sizeof(size_t), MPI_CHAR,
                    &lengths_in_[0], fnum_ * sizeof(size_t), MPI_CHAR,
                    comm_spec_.comm());
#ifdef PROFILING
      marker.Stop();
#endif
    }

    return std::all_of(lengths_in_.begin(), lengths_in_.end(),
                       [](size_t size) { return size == 0; });
  }

  InArchiveGroup to_send_;
  OutArchiveGroup to_recv_;
  pinned_vector<dev::OutArchive> pinned_to_recv_;
  thrust::device_vector<dev::InArchive> d_to_send_;
  thrust::device_vector<dev::OutArchive> d_to_recv_;

  Event computation_finished_;

  grape::CommSpec comm_spec_;
  std::shared_ptr<ncclComm_t> nccl_comm_;

  Stream compute_stream_;
  Stream comm_stream_;

  std::vector<size_t> lengths_out_;
  std::vector<size_t> lengths_in_;

  size_t sent_size_{};
  double total_memcpy_time_{};

  bool to_terminate_{};
  bool force_continue_{};

  fid_t fid_;
  fid_t fnum_;
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_PARALLEL_GPU_MESSAGE_MANAGER_H_
