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

#ifndef GRAPE_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_
#define GRAPE_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_

#include <memory>
#include <thread>
#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/utils/vertex_array.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief A kind of collective message manager.
 *
 * This message manager is designed for the scenario that all mirror vertices'
 * state need to be override with their masters' state, e.g. PageRank.
 *
 * After a round, message manager will encode the inner vertices' state of a
 * vertex array for each other fragment.
 *
 * When receive a batch of messages, message manager will update the state of
 * outer vertices in a designated vertex array.
 */
class BatchShuffleMessageManager : public MessageManagerBase {
 public:
  BatchShuffleMessageManager() : comm_(NULL_COMM) {}
  ~BatchShuffleMessageManager() {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  /**
   * @brief Inherit
   */
  void Init(MPI_Comm comm) {
    MPI_Comm_dup(comm, &comm_);

    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    shuffle_out_buffers_.resize(fnum_);

    recv_thread_ =
        std::thread(&BatchShuffleMessageManager::recvThreadRoutine, this);
  }

  /**
   * @brief Inherit
   */
  void Start() {}

  /**
   * @brief Inherit
   */
  void StartARound() {
    msg_size_ = 0;
    to_terminate_ = true;
  }

  /**
   * @brief Inherit
   */
  void FinishARound() {}

  /**
   * @brief Inherit
   */
  bool ToTerminate() { return to_terminate_; }

  /**
   * @brief Inherit
   */
  size_t GetMsgSize() const { return msg_size_; }

  /**
   * @brief Inherit
   */
  void Finalize() {
    if (!send_reqs_.empty()) {
      MPI_Waitall(send_reqs_.size(), &send_reqs_[0], MPI_STATUSES_IGNORE);
      send_reqs_.clear();
    }

    if (!recv_reqs_.empty()) {
      MPI_Waitall(recv_reqs_.size(), &recv_reqs_[0], MPI_STATUSES_IGNORE);
      recv_reqs_.clear();
    }

    {
      size_t v = 1;
      MPI_Send(&v, sizeof(size_t), MPI_CHAR, comm_spec_.FragToWorker(fid_), 1,
               comm_);
      recv_thread_.join();
    }

    MPI_Comm_free(&comm_);
    comm_ = NULL_COMM;
  }

  /**
   * @brief Synchronize the inner vertices' data of a vertex array to their
   * mirrors.
   *
   * @tparam GRAPH_T
   * @tparam DATA_T
   * @param frag
   * @param data_out The inner vertices data of data_out will be sent.
   * @param data_in The outer vertices data of data_in will be updated.
   */
  template <typename GRAPH_T, typename DATA_T>
  void SyncInnerVertices(
      const GRAPH_T& frag,
      const VertexArray<DATA_T, typename GRAPH_T::vid_t>& data_out,
      VertexArray<DATA_T, typename GRAPH_T::vid_t>& data_in,
      int thread_num = std::thread::hardware_concurrency()) {
    to_terminate_ = false;

    if (!send_reqs_.empty()) {
      MPI_Waitall(send_reqs_.size(), &send_reqs_[0], MPI_STATUSES_IGNORE);
      send_reqs_.clear();
    }

    if (!recv_reqs_.empty()) {
      MPI_Waitall(recv_reqs_.size(), &recv_reqs_[0], MPI_STATUSES_IGNORE);
      recv_reqs_.clear();
      recv_from_.clear();
    }

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto range = frag.OuterVertices(src_fid);
      MPI_Request req;
      MPI_Irecv(&data_in[range.begin()], range.size() * sizeof(DATA_T),
                MPI_CHAR, comm_spec_.FragToWorker(src_fid), 0, comm_, &req);
      recv_reqs_.push_back(req);
      recv_from_.push_back(src_fid);
    }

    remaining_reqs_ = fnum_ - 1;

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (i + fid_) % fnum_;
      auto& id_vec = frag.MirrorVertices(dst_fid);
      auto& vec = shuffle_out_buffers_[dst_fid];
      vec.clear();
      vec.resize(id_vec.size() * sizeof(DATA_T));
      DATA_T* buf = reinterpret_cast<DATA_T*>(vec.data());
      size_t num = id_vec.size();
#pragma omp parallel for num_threads(thread_num)
      for (size_t k = 0; k < num; ++k) {
        buf[k] = data_out[id_vec[k]];
      }

      MPI_Request req;
      MPI_Isend(vec.data(), vec.size(), MPI_CHAR,
                comm_spec_.FragToWorker(dst_fid), 0, comm_, &req);
      msg_size_ += vec.size();
      send_reqs_.push_back(req);
    }
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
  void SyncInnerVertices(const GRAPH_T& frag,
                         VertexArray<DATA_T, typename GRAPH_T::vid_t>& data,
                         int thread_num = std::thread::hardware_concurrency()) {
    to_terminate_ = false;

    if (!send_reqs_.empty()) {
      MPI_Waitall(send_reqs_.size(), &send_reqs_[0], MPI_STATUSES_IGNORE);
      send_reqs_.clear();
    }

    if (!recv_reqs_.empty()) {
      MPI_Waitall(recv_reqs_.size(), &recv_reqs_[0], MPI_STATUSES_IGNORE);
      recv_reqs_.clear();
      recv_from_.clear();
    }

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto range = frag.OuterVertices(src_fid);
      MPI_Request req;
      MPI_Irecv(&data[range.begin()], range.size() * sizeof(DATA_T), MPI_CHAR,
                comm_spec_.FragToWorker(src_fid), 0, comm_, &req);
      recv_reqs_.push_back(req);
      recv_from_.push_back(src_fid);
    }

    remaining_reqs_ = fnum_ - 1;

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (i + fid_) % fnum_;
      auto& id_vec = frag.MirrorVertices(dst_fid);
      auto& vec = shuffle_out_buffers_[dst_fid];
      vec.clear();
      vec.resize(id_vec.size() * sizeof(DATA_T));
      DATA_T* buf = reinterpret_cast<DATA_T*>(vec.data());
      size_t num = id_vec.size();
#pragma omp parallel for num_threads(thread_num)
      for (size_t k = 0; k < num; ++k) {
        buf[k] = data[id_vec[k]];
      }

      MPI_Request req;
      MPI_Isend(vec.data(), vec.size(), MPI_CHAR,
                comm_spec_.FragToWorker(dst_fid), 0, comm_, &req);
      msg_size_ += vec.size();
      send_reqs_.push_back(req);
    }
  }

  /**
   * @brief This function will block until all outer vertices are updated, that
   * is, messages from all other fragments are received.
   */
  void UpdateOuterVertices() {
    MPI_Waitall(recv_reqs_.size(), &recv_reqs_[0], MPI_STATUSES_IGNORE);
    remaining_reqs_ = 0;
    recv_reqs_.clear();
    recv_from_.clear();
  }

  /**
   * @brief This function will block until a set of messages from one fragment
   * are received.
   *
   * @return Source fragment id.
   */
  fid_t UpdatePartialOuterVertices() {
    int index;
    fid_t ret;
    MPI_Waitany(recv_reqs_.size(), &recv_reqs_[0], &index, MPI_STATUS_IGNORE);
    remaining_reqs_--;
    ret = recv_from_[index];
    if (remaining_reqs_ == 0) {
      recv_reqs_.clear();
      recv_from_.clear();
    }
    return ret;
  }

  /**
   * @brief Inherit
   */
  void ForceContinue() {}

 private:
  void recvThreadRoutine() {
    std::vector<MPI_Request> recv_thread_reqs(fnum_);
    std::vector<size_t> numbers(fnum_);
    for (fid_t src_fid = 0; src_fid < fnum_; ++src_fid) {
      MPI_Irecv(&numbers[src_fid], sizeof(size_t), MPI_CHAR,
                comm_spec_.FragToWorker(src_fid), 1, comm_,
                &recv_thread_reqs[src_fid]);
    }
    int index;
    MPI_Waitany(fnum_, &recv_thread_reqs[0], &index, MPI_STATUS_IGNORE);
    CHECK(index == static_cast<int>(fid_));
    for (fid_t src_fid = 0; src_fid < fnum_; ++src_fid) {
      if (src_fid != fid_) {
        MPI_Cancel(&recv_thread_reqs[src_fid]);
      }
    }
  }

  fid_t fid_;
  fid_t fnum_;
  CommSpec comm_spec_;

  MPI_Comm comm_;

  std::vector<std::vector<char>> shuffle_out_buffers_;

  std::vector<MPI_Request> recv_reqs_;
  std::vector<fid_t> recv_from_;
  fid_t remaining_reqs_;

  std::vector<MPI_Request> send_reqs_;

  size_t msg_size_;
  std::thread recv_thread_;

  bool to_terminate_;
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_
