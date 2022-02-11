/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef GRAPE_PARALLEL_DEFAULT_MESSAGE_MANAGER_H_
#define GRAPE_PARALLEL_DEFAULT_MESSAGE_MANAGER_H_

#include <memory>
#include <utility>
#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief Default message manager.
 *
 * The send and recv methods are not thread-safe.
 */
class DefaultMessageManager : public MessageManagerBase {
 public:
  DefaultMessageManager() : comm_(NULL_COMM) {}
  ~DefaultMessageManager() override {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  /**
   * @brief Inherit
   */
  void Init(MPI_Comm comm) override {
    MPI_Comm_dup(comm, &comm_);

    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    force_terminate_ = false;
    terminate_info_.Init(fnum_);

    lengths_out_.resize(fnum_);
    lengths_in_.resize(fnum_ * fnum_);

    to_send_.resize(fnum_);
    to_recv_.resize(fnum_);
  }

  /**
   * @brief Inherit
   */
  void Start() override {}

  /**
   * @brief Inherit
   */
  void StartARound() override {
    sent_size_ = 0;
    if (!reqs_.empty()) {
      MPI_Waitall(reqs_.size(), &reqs_[0], MPI_STATUSES_IGNORE);
      reqs_.clear();
    }
    for (auto& arc : to_send_) {
      arc.Clear();
    }
    force_continue_ = false;
    cur_ = 0;
  }

  /**
   * @brief Inherit
   */
  void FinishARound() override {
    to_terminate_ = syncLengths();
    if (to_terminate_) {
      return;
    }

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + i) % fnum_;
      size_t length = lengths_in_[src_fid * fnum_ + fid_];
      if (length == 0) {
        continue;
      }
      auto& arc = to_recv_[src_fid];
      arc.Clear();
      arc.Allocate(length);
      sync_comm::irecv_buffer<char>(arc.GetBuffer(), length,
                                    comm_spec_.FragToWorker(src_fid), 0, comm_,
                                    reqs_);
    }

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (fid_ + fnum_ - i) % fnum_;
      auto& arc = to_send_[dst_fid];
      if (arc.Empty()) {
        continue;
      }
      sync_comm::isend_buffer<char>(arc.GetBuffer(), arc.GetSize(),
                                    comm_spec_.FragToWorker(dst_fid), 0, comm_,
                                    reqs_);
    }
    to_recv_[fid_].Clear();
    if (!to_send_[fid_].Empty()) {
      to_recv_[fid_] = std::move(to_send_[fid_]);
    }
  }

  /**
   * @brief Inherit
   */
  bool ToTerminate() override { return to_terminate_; }

  /**
   * @brief Inherit
   */
  void Finalize() override {
    if (!reqs_.empty()) {
      MPI_Waitall(reqs_.size(), &reqs_[0], MPI_STATUSES_IGNORE);
      reqs_.clear();
    }

    MPI_Comm_free(&comm_);
    comm_ = NULL_COMM;
  }

  /**
   * @brief Inherit
   */
  size_t GetMsgSize() const override { return sent_size_; }

  /**
   * @brief Inherit
   */
  void ForceContinue() override { force_continue_ = true; }

  /**
   * @brief Inherit
   */
  void ForceTerminate(const std::string& terminate_info) override {
    force_terminate_ = true;
    terminate_info_.info[comm_spec_.fid()] = terminate_info;
  }

  /**
   * @brief Inherit
   */
  const TerminateInfo& GetTerminateInfo() const override {
    return terminate_info_;
  }

  /**
   * @brief Send message to a fragment.
   *
   * @tparam MESSAGE_T Message type.
   * @param dst_fid Destination fragment id.
   * @param msg
   */
  template <typename MESSAGE_T>
  inline void SendToFragment(fid_t dst_fid, const MESSAGE_T& msg) {
    to_send_[dst_fid] << msg;
  }

  /**
   * @brief Communication by synchronizing the status on outer vertices, for
   * edge-cut fragments.
   *
   * Assume a fragment F_1, a crossing edge a->b' in F_1 and a is an inner
   * vertex in F_1. This function invoked on F_1 send status on b' to b on F_2,
   * where b is an inner vertex.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SyncStateOnOuterVertex(const GRAPH_T& frag,
                                     const typename GRAPH_T::vertex_t& v,
                                     const MESSAGE_T& msg) {
    fid_t fid = frag.GetFragId(v);
    to_send_[fid] << frag.GetOuterVertexGid(v) << msg;
  }

  /**
   * @brief Communication via a crossing edge a<-c. It sends message
   * from a to c.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughIEdges(const GRAPH_T& frag,
                                   const typename GRAPH_T::vertex_t& v,
                                   const MESSAGE_T& msg) {
    auto dsts = frag.IEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid] << gid << msg;
    }
  }

  /**
   * @brief Communication via a crossing edge a->b. It sends message
   * from a to b.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughOEdges(const GRAPH_T& frag,
                                   const typename GRAPH_T::vertex_t& v,
                                   const MESSAGE_T& msg) {
    auto dsts = frag.OEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid] << gid << msg;
    }
  }

  /**
   * @brief Communication via crossing edges a->b and a<-c. It sends message
   * from a to b and c.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v: a
   * @param msg
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline void SendMsgThroughEdges(const GRAPH_T& frag,
                                  const typename GRAPH_T::vertex_t& v,
                                  const MESSAGE_T& msg) {
    auto dsts = frag.IOEDests(v);
    const fid_t* ptr = dsts.begin;
    typename GRAPH_T::vid_t gid = frag.GetInnerVertexGid(v);
    while (ptr != dsts.end) {
      fid_t fid = *(ptr++);
      to_send_[fid] << gid << msg;
    }
  }

  /**
   * @brief Get a message from message buffer.
   *
   * @tparam MESSAGE_T
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename MESSAGE_T>
  inline bool GetMessage(MESSAGE_T& msg) {
    while (cur_ != fnum_ && to_recv_[cur_].Empty()) {
      ++cur_;
    }
    if (cur_ == fnum_) {
      return false;
    }
    to_recv_[cur_] >> msg;
    return true;
  }

  /**
   * @brief Get a message and its target vertex from message buffer.
   *
   * @tparam GRAPH_T
   * @tparam MESSAGE_T
   * @param frag
   * @param v
   * @param msg
   *
   * @return Return true if got a message, and false if no message left.
   */
  template <typename GRAPH_T, typename MESSAGE_T>
  inline bool GetMessage(const GRAPH_T& frag, typename GRAPH_T::vertex_t& v,
                         MESSAGE_T& msg) {
    while (cur_ != fnum_ && to_recv_[cur_].Empty()) {
      ++cur_;
    }
    if (cur_ == fnum_) {
      return false;
    }
    typename GRAPH_T::vid_t gid;
    to_recv_[cur_] >> gid >> msg;
    frag.Gid2Vertex(gid, v);
    return true;
  }

 protected:
  fid_t fid() const { return fid_; }
  fid_t fnum() const { return fnum_; }

  std::vector<InArchive> to_send_;

 private:
  bool syncLengths() {
    for (fid_t i = 0; i < fnum_; ++i) {
      sent_size_ += to_send_[i].GetSize();
      lengths_out_[i] = to_send_[i].GetSize();
    }
    if (force_continue_) {
      ++lengths_out_[fid_];
    }
    int terminate_flag = force_terminate_ ? 1 : 0;
    int terminate_flag_sum;
    MPI_Allreduce(&terminate_flag, &terminate_flag_sum, 1, MPI_INT, MPI_SUM,
                  comm_);
    if (terminate_flag_sum > 0) {
      terminate_info_.success = false;
      sync_comm::AllGather(terminate_info_.info, comm_);
      return true;
    } else {
      MPI_Allgather(&lengths_out_[0], fnum_ * sizeof(size_t), MPI_CHAR,
                    &lengths_in_[0], fnum_ * sizeof(size_t), MPI_CHAR, comm_);
      for (auto s : lengths_in_) {
        if (s != 0) {
          return false;
        }
      }
      return true;
    }
  }

  std::vector<OutArchive> to_recv_;
  fid_t cur_;

  std::vector<size_t> lengths_out_;
  std::vector<size_t> lengths_in_;

  std::vector<MPI_Request> reqs_;
  MPI_Comm comm_;

  fid_t fid_;
  fid_t fnum_;
  CommSpec comm_spec_;

  size_t sent_size_;
  bool to_terminate_;
  bool force_continue_;
  bool force_terminate_;

  TerminateInfo terminate_info_;
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_DEFAULT_MESSAGE_MANAGER_H_
