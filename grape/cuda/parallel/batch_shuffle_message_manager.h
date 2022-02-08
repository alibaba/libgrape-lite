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

#ifndef GRAPE_CUDA_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_
#define GRAPE_CUDA_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_

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
#include "grape/cuda/utils/stream.h"
#include "grape/cuda/utils/vertex_array.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/util.h"

namespace grape {
namespace cuda {
class BatchShuffleMessageManager {
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

    shuffle_in_buffers_.resize(fnum_);
    shuffle_out_buffers_.resize(fnum_);

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
    {
      size_t length = 4 * 1024 * 1024;
      std::vector<thrust::device_vector<char>> buf_in(fnum_), buf_out(fnum_);

      CHECK_NCCL(ncclGroupStart());

      for (fid_t i = 1; i < fnum_; ++i) {
        fid_t src_fid = (fid_ + i) % fnum_;
        int peer = comm_spec_.FragToWorker(src_fid);

        buf_in[src_fid].resize(length);
        CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(buf_in[src_fid].data()),
                            length, ncclChar, peer, *nccl_comm_,
                            comm_stream_.cuda_stream()));
      }

      for (fid_t i = 1; i < fnum_; ++i) {
        fid_t dst_fid = (fid_ + fnum_ - i) % fnum_;
        int peer = comm_spec_.FragToWorker(dst_fid);

        buf_out[dst_fid].resize(length);
        CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(buf_out[dst_fid].data()),
                            length, ncclChar, peer, *nccl_comm_,
                            comm_stream_.cuda_stream()));
      }

      CHECK_NCCL(ncclGroupEnd());
    }
    comm_stream_.Sync();
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
    to_terminate_ = true;
    sent_size_ = 0;
    computation_finished_ = Event::Create();
  }
  /**
   * @brief This function will be called after each evaluation step of
   * applications.
   */
  void FinishARound() {}
  /**
   * @brief This function will be called after the evaluation of applications.
   */
  void Finalize() const {
    VLOG(2) << "Memory copy time: " << total_memcpy_time_ * 1000 << " ms";
  }

  /**
   * @brief Synchronize the inner vertices' data of a vertex array to their
   * mirrors. The data_out and data_in are the same vertex array.
   *
   * @tparam GRAPH_T
   * @tparam DATA_T
   * @param frag
   * @param data
   */
  template <typename GRAPH_T, typename DATA_T>
  void SyncInnerVertices(const GRAPH_T& h_frag,
                         VertexArray<DATA_T, typename GRAPH_T::vid_t>& h_data) {
    to_terminate_ = false;
    computation_finished_.Record(compute_stream_);
    computation_finished_.Wait(comm_stream_);

    double memcpy_time = grape::GetCurrentTime();

    auto d_frag = h_frag.DeviceObject();
    auto d_data = h_data.DeviceObject();

    CHECK_NCCL(ncclGroupStart());

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (i + fid_) % fnum_;
      int peer = comm_spec_.FragToWorker(src_fid);
      auto& vec = shuffle_in_buffers_[src_fid];

      shuffle_in_buffers_[src_fid].resize(h_frag.OuterVertices(src_fid).size() *
                                          sizeof(DATA_T));

      CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(vec.data()), vec.size(),
                          ncclChar, peer, *nccl_comm_,
                          comm_stream_.cuda_stream()));
    }

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (fid_ + fnum_ - i) % fnum_;
      int peer = comm_spec_.FragToWorker(dst_fid);
      auto& vec = shuffle_out_buffers_[dst_fid];
      auto nv_mirror = h_frag.MirrorVertices(dst_fid).size();

      vec.resize(nv_mirror * sizeof(DATA_T));

      // Memcpy with launching kernel
      LaunchKernel(
          comm_stream_, nv_mirror,
          [=] __device__(DATA_T * buf) {
            auto tid = TID_1D;
            auto nthreads = TOTAL_THREADS_1D;
            auto id_vec = d_frag.MirrorVertices(dst_fid);

            for (size_t i = 0 + tid; i < id_vec.size(); i += nthreads) {
              buf[i] = d_data[id_vec[i]];
            }
          },
          reinterpret_cast<DATA_T*>(thrust::raw_pointer_cast(vec.data())));

      CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(vec.data()), vec.size(),
                          ncclChar, peer, *nccl_comm_,
                          comm_stream_.cuda_stream()));
      sent_size_ += vec.size();
    }
    CHECK_NCCL(ncclGroupEnd());

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (i + fid_) % fnum_;
      auto& vec = shuffle_in_buffers_[src_fid];

      // N.B. NCCL does not support sending and receiving data with different
      // types, so we need a copy
      CHECK_CUDA(cudaMemcpyAsync(
          thrust::raw_pointer_cast(d_data.data()) +
              h_frag.OuterVertices(src_fid).begin().GetValue(),
          thrust::raw_pointer_cast(vec.data()), vec.size(),
          cudaMemcpyDeviceToDevice, comm_stream_.cuda_stream()));
    }
    comm_stream_.Sync();

    float size_in_mb = static_cast<float>(sent_size_) / 1024 / 1024;
    memcpy_time = grape::GetCurrentTime() - memcpy_time;

    VLOG(2) << "Worker " << fid_ << " Copy size " << size_in_mb << " MB, "
            << "Time: " << memcpy_time * 1000 << " ms, "
            << "Bandwith: " << size_in_mb / memcpy_time << " MB/S";

    total_memcpy_time_ += memcpy_time;
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
  void ForceContinue() { to_terminate_ = false; }

  Stream& stream() { return compute_stream_; }

  ncclComm_t nccl_comm() { return *nccl_comm_; }

  double GetAccumulatedCommTime() const { return total_memcpy_time_; }

 private:
  grape::CommSpec comm_spec_;
  fid_t fid_;
  fid_t fnum_;
  std::shared_ptr<ncclComm_t> nccl_comm_;
  Stream compute_stream_;
  Stream comm_stream_;

  std::vector<thrust::device_vector<char>> shuffle_in_buffers_;
  std::vector<thrust::device_vector<char>> shuffle_out_buffers_;

  Event computation_finished_;

  size_t sent_size_{};
  double total_memcpy_time_{};

  bool to_terminate_{};
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_
