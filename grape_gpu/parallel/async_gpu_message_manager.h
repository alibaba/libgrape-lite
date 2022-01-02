#ifndef GRAPE_GPU_PARALLEL_ASYNC_GPU_MESSAGE_MANAGER_H_
#define GRAPE_GPU_PARALLEL_ASYNC_GPU_MESSAGE_MANAGER_H_

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
#include "grape_gpu/parallel/async_message_kernels.h"
#include "grape_gpu/serialization/async_in_archive.h"
#include "grape_gpu/serialization/async_out_archive.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/event.h"
#include "grape_gpu/utils/markers.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/communication/driver.h"

namespace grape_gpu {
namespace dev {

template<typename S>
class AsyncMessageManager {
 public:
  explicit AsyncMessageManager(const ArrayView<AsyncInArchive<S>>& to_send)
      : to_send_(to_send) {}

  // We can not use warp aggregation for here, because the different InArchive
  // objects will be used
  // TODO: bugs in instantiation for dependent names
  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                         const typename GRAPH_T::vertex_t& v,
                                         const MESSAGE_T& msg) {
    using vid_t = typename GRAPH_T::vid_t;
    using data_t = thrust::pair<vid_t, MESSAGE_T>;
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].template AddBytes<data_t>(thrust::make_pair(frag.GetOuterVertexGid(v), msg));
  }

  template <typename GRAPH_T>
  DEV_INLINE void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                         const typename GRAPH_T::vertex_t& v) {
    using vid_t = typename GRAPH_T::vid_t;
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].template AddBytes<vid_t>(frag.GetOuterVertexGid(v));
  }

  // TODO: bugs in instantiation for dependent names
  template <typename GRAPH_T, typename MESSAGE_T>
  DEV_INLINE void SyncStateOnOuterVertexWarpOpt(
      const GRAPH_T& frag, const typename GRAPH_T::vertex_t& v,
      const MESSAGE_T& msg) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytesWarp(
        thrust::make_pair(frag.GetOuterVertexGid(v), msg));
  }

  // TODO: bugs in instantiation for dependent names
  template <typename GRAPH_T>
  DEV_INLINE void SyncStateOnOuterVertexWarpOpt(
      const GRAPH_T& frag, const typename GRAPH_T::vertex_t& v) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid].AddBytesWarp(frag.GetOuterVertexGid(v));
  }

  // TODO: bugs in instantiation for dependent names
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

  // TODO: bugs in instantiation for dependent names
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

  // TODO: bugs in instantiation for dependent names
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

  // TODO: bugs in instantiation for dependent names
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

  // TODO: bugs in instantiation for dependent names
  template <typename MESSAGE_T>
  DEV_INLINE void SendToFragment(fid_t dst_fid, const MESSAGE_T& msg) {
    to_send_[dst_fid].AddBytes(msg);
  }

  // TODO: bugs in instantiation for dependent names
  template <typename MESSAGE_T>
  DEV_INLINE void SendToFragmentWarpOpt(fid_t dst_fid, const MESSAGE_T& msg) {
    to_send_[dst_fid].AddBytesWarp(msg);
  }

 private:
  ArrayView<dev::AsyncInArchive<S>> to_send_;
};
}  // namespace dev

template<typename S>
class AsyncGPUMessageManager {
 public:
  AsyncGPUMessageManager() {}

  /**
   * @brief Initialize message manager.
   *
   * @param comm MPI_Comm object.
   */
  void Init(const grape::CommSpec& comm_spec) {
    comm_spec_ = comm_spec;
    fnum_ = comm_spec.fnum();
    fid_ = comm_spec.fid();
  }


  void InitDriver(size_t chunk_size, size_t chunk_num, size_t capacity) {
    driver_.Init(comm_spec_, chunk_size, chunk_num); 
    d_to_send_.resize(fnum_);
    for (fid_t fid = 0; fid < fnum_; fid++) {
      if (fid != fid_) {
        to_send_.emplace_back(capacity);
        to_recv_.emplace_back(capacity);
      } else {
        to_send_.emplace_back(0);
        to_recv_.emplace_back(0);
      }
      d_to_send_[fid] = to_send_[fid].DeviceObject();
    }
  }

  /** 
   * @brief This function will get message from driver, then put it into 
   * out archive.
   */
  void ProcessIncomingMsg() {
    for (size_t r = 0; r < driver_.Size(); ++r) {
      //if(fid_==0) VLOG(2) << "caosdofafu " << r;
      if(r == fid_) continue;
      bool hasmsg = driver_.Query(r);
      if (hasmsg) VLOG(2) << "message in " << r;
      if (r == fid_ || !hasmsg) continue;
      auto raw = driver_.ReceiveFrom(r, comm_stream_); // get ptr
      VLOG(2) << "found raw ptr" << raw.GetTotalSize();
      if (0 == raw.GetTotalSize()) continue;
      if(fid_==0) VLOG(2) << "do copy";
      to_recv_[r].PourSeg(raw, comm_stream_); // copy
      if(fid_==0) VLOG(2) << "return ptr";
      driver_.Restore(raw, comm_stream_);  // return ptr
      if(fid_==0) VLOG(2) << "over ";
    }
  }

  /** 
   * @brief This function will get message from in archive, then use IPCdirve to 
   * write into a remote target GPU directly. 
   */
  template<typename MESSAGE_T>
  void ProcessOutGoingMsg() {
    for (fid_t fid = 0; fid < fnum_; fid++) {
      if(fid == fid_) continue;
      auto segs = to_recv_[fid].template FetchSegs<MESSAGE_T>(comm_stream_); // get msg
      size_t size = 0;
      for(auto& s: segs){
        MESSAGE_T* msg = s.GetSegmentPtr();
        size_t msg_size = s.GetTotalSize();
        int sh = driver_.template SendTo<MESSAGE_T>(fid, msg, msg_size, comm_stream_);
        driver_.WaitSend(sh, comm_stream_);
        size += msg_size;
      }
      to_recv_[fid].PopAsync(comm_stream_, size);// popAsync
    }
  }

  /**
   * @brief This function will be called after the evaluation of applications.
   */
  void Finalize() const {}

  // TODO:
  inline void Register(int works) { need_to_process = works; }

  void Barrier() {
    MPI_Barrier(comm_spec_.comm());
  }

  bool IsSendEmpty () {
    bool ret = true;
    for(fid_t fid = 0; fid < fnum_; fid++) {
      if(fid == fid_) continue;
      ret &= to_send_[fid].Empty(comm_stream_);
    }
    return ret;
  }

  bool IsRecvEmpty() {
    bool ret = true;
    for(fid_t fid = 0; fid < fnum_; fid++) {
      if(fid == fid_) continue;
      ret &= to_recv_[fid].Empty(comm_stream_);
    }
    return ret;
  }

  bool IsCompEmpty() { return need_to_process == 0; }

  bool VoteToTerminate () { // comm_thread 
    if(!IsSendEmpty() || !IsRecvEmpty()) return false;
    // block until all send op are done
    MPI_Barrier(comm_spec_.comm());
    // check whether we still have messages to process;
    bool done = IsRecvEmpty();
    MPI_Allreduce(MPI_IN_PLACE, &done, 1, MPI_UNSIGNED_CHAR, MPI_LAND, comm_spec_.comm());
    // fall back to normal work;
    return done;
  }

  dev::AsyncMessageManager<S> DeviceObject() {
    return dev::AsyncMessageManager<S>(
              ArrayView<dev::AsyncInArchive<S>>(d_to_send_));
  }

  // consume pcqueue
  template <typename GRAPH_T, typename MESSAGE_T = grape::EmptyType,
            typename FUNC_T>
  inline void ParallelProcess(const GRAPH_T& frag, FUNC_T func) {
    int grid_size, block_size;

    for (fid_t fid = 0; fid < fnum_; fid++) {
      auto& recv = to_recv_[fid];

      //KernelSizing(
      //    grid_size, block_size,
      //    round_up(recv.AvailableBytes(comp_stream_),
      //             (sizeof(MESSAGE_T) + sizeof(typename GRAPH_T::vid_t))));
      grid_size = 256, block_size = 256;
      kernel::ProcessMsg<GRAPH_T, MESSAGE_T, FUNC_T, S>
          <<<grid_size, block_size, 0, comp_stream_.cuda_stream()>>>(
              recv.DeviceObject(), frag, func);
    }
  }

  template <typename MESSAGE_T = grape::EmptyType, typename FUNC_T>
  inline void ParallelProcess(FUNC_T func) {
    int grid_size, block_size;

    for (fid_t fid = 0; fid < fnum_; fid++) {
      auto& recv = to_recv_[fid];

      //KernelSizing(
      //    grid_size, block_size,
      //    round_up(recv.AvailableBytes(comp_stream_), sizeof(MESSAGE_T)));
      grid_size = 256, block_size = 256;
      kernel::ProcessMsg<MESSAGE_T, FUNC_T, S>
          <<<grid_size, block_size, 0, comp_stream_.cuda_stream()>>>(
              recv.DeviceObject(), func);
    }
  }

  Stream& stream() { return comp_stream_; }

 private:
  grape::CommSpec comm_spec_;
  Driver<ASYNC, char> driver_;
  fid_t fid_;
  fid_t fnum_;
  Stream comp_stream_;
  Stream comm_stream_;
  std::vector<AsyncInArchive<S>> to_send_;
  thrust::device_vector<dev::AsyncInArchive<S>> d_to_send_;
  std::vector<AsyncOutArchive<S>> to_recv_;
  int32_t need_to_process;
};

}  // namespace grape_gpu

#endif  // GRAPE_GPU_PARALLEL_GPU_ASYNC_MESSAGE_MANAGER_H_
