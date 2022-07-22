#ifndef GRAPE_GPU_PARALLEL_GPU_MESSAGE_MANAGER_H_
#define GRAPE_GPU_PARALLEL_GPU_MESSAGE_MANAGER_H_

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

#include "grape/parallel/message_manager_base.h"
#include "grape/util.h"
#include "grape_gpu/graph/adj_list.h"
#include "grape_gpu/parallel/message_kernels.h"
#include "grape_gpu/serialization/in_archive.h"
#include "grape_gpu/serialization/out_archive.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/event.h"
#include "grape_gpu/utils/markers.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/time_table.h"

DECLARE_bool(fuse);

namespace grape_gpu {
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
    to_send_[fid].AddBytesWarp(
        thrust::make_pair(frag.GetOuterVertexGid(v), msg));
  }

  template <typename GRAPH_T>
  DEV_INLINE void SyncStateOnOuterVertexWarpOpt(
      const GRAPH_T& frag, const typename GRAPH_T::vertex_t& v) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytesWarp(frag.GetOuterVertexGid(v));
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SendMsgThroughIEdges(const GRAPH_T& frag,
                                       const typename GRAPH_T::vertex_t& v,
                                       const MESSAGE_T& msg) {
    DestList dsts = frag.IEDests(v);
    fid_t* ptr = dsts.begin;
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
    fid_t* ptr = dsts.begin;
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
    fid_t* ptr = dsts.begin;
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
    fid_t* ptr = dsts.begin;
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
  DEV_INLINE void SendToFragment(fid_t dst_fid, uint32_t byte_pos,
                                 const MESSAGE_T& msg) {
    to_send_[dst_fid].AddBytes(byte_pos, msg);
  }

  template <typename MESSAGE_T>
  DEV_INLINE void SendToFragmentWarpOpt(fid_t dst_fid, const MESSAGE_T& msg) {
    to_send_[dst_fid].AddBytesWarp(msg);
  }

  template <typename MESSAGE_T>
  void Write(fid_t fid, size_t elem_pos, const MESSAGE_T& msg) {
    to_send_[fid].template AddBytes(elem_pos, msg);
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

    nccl_comm_ =
        std::shared_ptr<ncclComm_t>(new ncclComm_t, [](ncclComm_t* comm) {
          CHECK_NCCL(ncclCommDestroy(*comm));
          delete comm;
        });

    CHECK_NCCL(ncclCommInitRank(nccl_comm_.get(), comm_spec.worker_num(), id,
                                comm_spec.worker_id()));

    WarmupNccl(comm_spec, comm_stream_, nccl_comm_);
  }

  void InitBuffer(size_t send_buffer_capacity, size_t recv_buffer_capacity) {
    for (fid_t fid = 0; fid < fnum_; fid++) {
      to_send_.resize(fid, send_buffer_capacity);
      d_to_send_[fid] = to_send_.DeviceObject(fid);
      to_recv_.resize(fid, recv_buffer_capacity);
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
    to_rerun_ = false;
    sent_size_ = 0;

    computation_finished_ = Event::Create();
    to_send_.Clear(compute_stream_);
  }

  /**
   * @brief This function will be called after each evaluation step of
   * applications.
   */
  void FinishARound() {
    if(to_rerun_) {
      //time_sync_len_.push_back(0.0);
      //time_memcpy.push_back(0.0);
      return;
    }
    double syncLengths_time = grape::GetCurrentTime();
    to_terminate_ = syncLengths();

    to_recv_.Clear(comm_stream_);
    comm_stream_.Sync();
    syncLengths_time = grape::GetCurrentTime() - syncLengths_time;
    this->syncLengths_time_ += syncLengths_time;
    time_map_["Sync Len"].push_back(syncLengths_time * 1000);

    RangeMarker marker(true, "nccl calling");
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
    marker.Stop();

    auto self_len = lengths_out_[fid_];
    if (force_continue_ && self_len == 1) {
      self_len = 0;
    }

    if (self_len > 0) {
      // Handle self message
      to_recv_.resize(fid_, self_len);
      CHECK_CUDA(cudaMemcpyAsync(to_recv_.data(fid_), to_send_.data(fid_),
                                 self_len, cudaMemcpyDeviceToDevice,
                                 comm_stream_.cuda_stream()));
      pinned_to_recv_[fid_] = to_recv_.DeviceObject(fid_);
    }

    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_to_recv_.data()),
                               thrust::raw_pointer_cast(pinned_to_recv_.data()),
                               sizeof(dev::OutArchive) * fnum_,
                               cudaMemcpyHostToDevice,
                               comm_stream_.cuda_stream()));
    comm_stream_.Sync();

    float size_in_mb = (float) sent_size_ / 1024 / 1024;
    memcpy_time = grape::GetCurrentTime() - memcpy_time;

    total_memcpy_time_ += memcpy_time;
    time_map_["Memcpy"].push_back(memcpy_time * 1000);

    VLOG(2) << "Worker " << fid_ << " Copy size " << size_in_mb << " MB, "
            << "Time: " << memcpy_time * 1000 << " ms, "
            << "Acc Time: " << total_memcpy_time_ * 1000 << " ms, "
            << "Bandwith: " << size_in_mb / memcpy_time << " MB/S";
  }

  /**
   * @brief This function will be called after the evaluation of applications.
   */
  void Finalize() {
#ifdef PROFILING
    {
      TimeTable<double> tt(comm_spec_);

      for (auto& pair : time_map_) {
        if (!pair.second.empty()) {
          tt.AddColumn(pair.first, pair.second);
        }
      }

      tt.Print();
    }
    {
      TimeTable<size_t> tt(comm_spec_);

      for (auto& pair : counter_map_) {
        if (!pair.second.empty()) {
          tt.AddColumn(pair.first, pair.second);
        }
      }
      if (!counter_map_.empty()) {
        tt.Print();
      }
    }
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
  void ForceRerun() { to_rerun_ = true; }

  double GetAccumulatedCommTime() const { return total_memcpy_time_; }
  double GetSyncTime() const { return syncLengths_time_; }
  double GetStartARoundTime() const { return StartARound_time_; }
  double GetBarrierTime() const { return barrier_time_; }

  void RecordStarARoundTime(double change) {
    this->StartARound_time_ += change;
    //    time_map_["Start_a_round"].push_back(change * 1000);
  }

  void RecordBarrierTime(double change) {
    this->barrier_time_ += change;
    //    time_map_["Barrier"].push_back(change*1000);
  }

  void RecordUnpackTime(double change) {
    time_map_["Unpack"].push_back(change * 1000);
  }

  void RecordComputeTime(double change) {
    time_map_["Compute"].push_back(change * 1000);
  }

  void RecordPackTime(double change) {
    time_map_["Pack"].push_back(change * 1000);
  }

  void RecordTime(const std::string& name, double change) {
    time_map_[name].push_back(change * 1000);
  }

  void RecordCount(const std::string& name, size_t count) {
    counter_map_[name].push_back(count);
  }

  dev::MessageManager DeviceObject() {
    return dev::MessageManager(ArrayView<dev::InArchive>(d_to_send_));
  }

  template <typename GRAPH_T, typename MESSAGE_T = grape::EmptyType,
            typename FUNC_T>
  inline void ParallelProcess(const GRAPH_T& frag, FUNC_T func) {
    int grid_size, block_size;

    RangeMarker marker(true, "ParallelProcess");
    if (fLB::FLAGS_fuse) {
      CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
          &grid_size, &block_size,
          kernel::ProcessMsgFused<GRAPH_T, MESSAGE_T, FUNC_T>, 0,
          (int) MAX_BLOCK_SIZE));

      kernel::ProcessMsgFused<GRAPH_T, MESSAGE_T, FUNC_T>
          <<<grid_size, block_size, 0, compute_stream_.cuda_stream()>>>(
              ArrayView<dev::OutArchive>(d_to_recv_), frag, func);
    } else {
      for (fid_t fid = 0; fid < fnum_; fid++) {
        auto length = to_recv_.AvailableBytes(fid);

        if (length > 0) {
          KernelSizing(grid_size, block_size,
                       round_up(length, (sizeof(MESSAGE_T) +
                                         sizeof(typename GRAPH_T::vid_t))));
          kernel::ProcessMsg<GRAPH_T, MESSAGE_T, FUNC_T>
              <<<grid_size, block_size, 0, compute_stream_.cuda_stream()>>>(
                  to_recv_.DeviceObject(fid), frag, func);
        }
      }
    }
    compute_stream_.Sync();
    marker.Stop();
  }

  template <typename MESSAGE_T = grape::EmptyType, typename FUNC_T>
  inline void ParallelProcess(FUNC_T func) {
    int grid_size, block_size;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &grid_size, &block_size, kernel::ProcessMsgFused<MESSAGE_T, FUNC_T>, 0,
        (int) MAX_BLOCK_SIZE));

    kernel::ProcessMsgFused<MESSAGE_T, FUNC_T>
        <<<grid_size, block_size, 0, compute_stream_.cuda_stream()>>>(
            ArrayView<dev::OutArchive>(d_to_recv_), func);

    compute_stream_.Sync();
  }

  Stream& stream() { return compute_stream_; }

  ncclComm_t nccl_comm() { return *nccl_comm_; }

  grape::CommSpec& comm_spec() { return comm_spec_; }

  template <typename FRAG_T, typename WORK_SOURCE_T, typename GEN_MSG>
  void MakeOutput(const Stream& stream, const FRAG_T& frag, WORK_SOURCE_T ws,
                  GEN_MSG gen_msg) {
    using vertex_t = typename FRAG_T::vertex_t;
    using msg_t = typename std::result_of<GEN_MSG&(vertex_t)>::type;
    auto fnum = frag.fnum();
    auto d_frag = frag.DeviceObject();
    auto* d_offset_sum = to_send_.get_size_ptr();
    int grid_size, block_size;
    auto d_mm = DeviceObject();

    KernelSizing(grid_size, block_size, ws.size());
    CHECK_GT(block_size, fnum);

    KernelWrapper<<<grid_size, block_size, fnum * sizeof(uint32_t),
                    stream.cuda_stream()>>>([=] __device__() mutable {
      typedef cub::BlockScan<uint32_t, MAX_BLOCK_SIZE> BlockScan;
      __shared__ typename BlockScan::TempStorage temp_storage;
      extern __shared__ char shared_mm_makeoutput[];
      auto* offset_sum = reinterpret_cast<uint32_t*>(&shared_mm_makeoutput[0]);
      auto size = ws.size();
      auto size_rup = round_up(size, blockDim.x) * blockDim.x;

      for (size_t i = TID_1D; i < size_rup; i += TOTAL_THREADS_1D) {
        fid_t send_to = fnum;
        uint32_t write_offset;
        vertex_t v;

        if (i < size) {
          v = ws.GetWork(i);
          send_to = d_frag.GetFragId(v);
        }

        for (fid_t fid = 0; fid < fnum; fid++) {
          uint32_t offset, vote = (send_to == fid ? sizeof(msg_t) : 0);

          BlockScan(temp_storage).ExclusiveSum(vote, offset);
          if (send_to == fid) {
            write_offset = offset;
          }

          if (threadIdx.x == blockDim.x - 1) {
            offset_sum[fid] = offset + vote;
          }
          __syncthreads();
        }

        if (threadIdx.x < fnum) {
          offset_sum[threadIdx.x] =
              atomicAdd(&d_offset_sum[threadIdx.x], offset_sum[threadIdx.x]);
        }
        __syncthreads();
        if (i < size) {
          write_offset += offset_sum[send_to];
          auto msg = gen_msg(v);
          d_mm.template SendToFragment(send_to, write_offset, msg);
        }
      }
    });
  }

  std::map<std::string, std::vector<double>> time_map_;
  std::map<std::string, std::vector<size_t>> counter_map_;

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
      RangeMarker marker(true, "MsgManager - Allgather");
      MPI_Allgather(&lengths_out_[0], fnum_, my_MPI_SIZE_T, &lengths_in_[0],
                    fnum_, my_MPI_SIZE_T, comm_spec_.comm());
      marker.Stop();
    }

    return std::all_of(lengths_in_.begin(), lengths_in_.end(),
                       [](size_t size) { return size == 0; });
  }

  ncclComm_t get_nccl_comm() { return *nccl_comm_; }

  grape::CommSpec comm_spec_;
  fid_t fid_;
  fid_t fnum_;
  std::shared_ptr<ncclComm_t> nccl_comm_;
  Stream compute_stream_;
  Stream comm_stream_;
  InArchiveGroup to_send_;
  thrust::device_vector<dev::InArchive> d_to_send_;
  OutArchiveGroup to_recv_;
  pinned_vector<dev::OutArchive> pinned_to_recv_;
  thrust::device_vector<dev::OutArchive> d_to_recv_;

  std::vector<size_t> lengths_out_;
  std::vector<size_t> lengths_in_;

  Event computation_finished_;

  size_t sent_size_{};
  double total_memcpy_time_{};
  double syncLengths_time_{};
  double StartARound_time_{};
  double barrier_time_{};

  bool to_rerun_{};
  bool to_terminate_{};
  bool force_continue_{};
};

}  // namespace grape_gpu

#endif  // GRAPE_GPU_PARALLEL_GPU_MESSAGE_MANAGER_H_
