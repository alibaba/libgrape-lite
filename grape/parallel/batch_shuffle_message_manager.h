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

namespace batch_shuffle_message_manager_impl {

template <typename FRAG_T>
struct IsRange {
  using sub_vertices_t = typename FRAG_T::sub_vertices_t;
  using vid_t = typename FRAG_T::vid_t;
  static constexpr bool value =
      std::is_same<sub_vertices_t, VertexRange<vid_t>>::value;
};

template <typename FRAG_T, typename MESSAGE_T>
struct ShuffleInplace {
  static constexpr bool value =
      IsRange<FRAG_T>::value && std::is_pod<MESSAGE_T>::value;
};

template <typename FRAG_T, typename MESSAGE_T>
struct PodShuffle {
  static constexpr bool value =
      !IsRange<FRAG_T>::value && std::is_pod<MESSAGE_T>::value;
};

template <typename MESSAGE_T>
struct ArchiveShuffle {
  static constexpr bool value = !std::is_pod<MESSAGE_T>::value;
};

class PostProcessBase {
 public:
  PostProcessBase() {}
  virtual ~PostProcessBase() {}

  virtual void exec(fid_t fid) = 0;
};

template <typename GRAPH_T, typename DATA_T>
class PostProcess : public PostProcessBase {
  using array_t = typename GRAPH_T::template vertex_array_t<DATA_T>;

 public:
  PostProcess(const GRAPH_T& frag, array_t& data,
              std::vector<std::vector<char>>& buffers)
      : frag_(frag), data_(data), buffers_(buffers) {}

  void exec(fid_t fid) {
    if (fid == frag_.fid()) {
      return;
    }

    auto& vec = buffers_[fid];
    OutArchive arc;
    arc.SetSlice(vec.data(), vec.size());
    auto& vertices = frag_.OuterVertices(fid);
    for (auto v : vertices) {
      arc >> data_[v];
    }
    buffers_[fid].clear();
  }

 private:
  const GRAPH_T& frag_;
  array_t& data_;
  std::vector<std::vector<char>>& buffers_;
};

}  // namespace batch_shuffle_message_manager_impl

class BatchShuffleMessageManager : public MessageManagerBase {
  template <typename FRAG_T, typename MESSAGE_T>
  using shuffle_inplace_t =
      batch_shuffle_message_manager_impl::ShuffleInplace<FRAG_T, MESSAGE_T>;

  template <typename FRAG_T, typename MESSAGE_T>
  using pod_shuffle_t =
      batch_shuffle_message_manager_impl::PodShuffle<FRAG_T, MESSAGE_T>;

  template <typename FRAG_T>
  using archive_shuffle_t =
      batch_shuffle_message_manager_impl::ArchiveShuffle<FRAG_T>;

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
  void Init(MPI_Comm comm) override {
    MPI_Comm_dup(comm, &comm_);

    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    remaining_reqs_.resize(fnum_);

    force_terminate_ = false;
    terminate_info_.Init(fnum_);

    shuffle_out_buffers_.resize(fnum_);
    shuffle_in_buffers_.resize(fnum_);

    recv_thread_ =
        std::thread(&BatchShuffleMessageManager::recvThreadRoutine, this);
  }

  /**
   * @brief Inherit
   */
  void Start() override {}

  /**
   * @brief Inherit
   */
  void StartARound() override {
    msg_size_ = 0;
    to_terminate_ = true;
  }

  /**
   * @brief Inherit
   */
  void FinishARound() override {}

  /**
   * @brief Inherit
   */
  bool ToTerminate() override {
    int flag = force_terminate_ ? 1 : 0;
    int ret;
    MPI_Allreduce(&flag, &ret, 1, MPI_INT, MPI_SUM, comm_);
    if (ret > 0) {
      terminate_info_.success = false;
      sync_comm::AllGather(terminate_info_.info, comm_);
      return true;
    }
    return to_terminate_;
  }

  /**
   * @brief Inherit
   */
  size_t GetMsgSize() const override { return msg_size_; }

  /**
   * @brief Inherit
   */
  void Finalize() override {
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
   * @param data
   */
  template <typename GRAPH_T, typename DATA_T>
  void SyncInnerVertices(
      const GRAPH_T& frag,
      typename GRAPH_T::template vertex_array_t<DATA_T>& data,
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

    startRecv<GRAPH_T, DATA_T>(frag, data, thread_num);

    remaining_frags_ = fnum_ - 1;

    startSend<GRAPH_T, DATA_T>(frag, data, thread_num);
  }

  /**
   * @brief This function will block until all outer vertices are updated, that
   * is, messages from all other fragments are received.
   */
  void UpdateOuterVertices() {
    MPI_Waitall(recv_reqs_.size(), &recv_reqs_[0], MPI_STATUSES_IGNORE);
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
    while (true) {
      MPI_Waitany(recv_reqs_.size(), &recv_reqs_[0], &index, MPI_STATUS_IGNORE);
      ret = recv_from_[index];
      --remaining_reqs_[ret];
      if (remaining_reqs_[ret] == 0) {
        --remaining_frags_;
        if (remaining_frags_ == 0) {
          recv_reqs_.clear();
          recv_from_.clear();
        }
        break;
      }
    }
    if (post_process_handle_ != nullptr) {
      post_process_handle_->exec(ret);
    }
    return ret;
  }

  /**
   * @brief Inherit
   */
  void ForceContinue() override {}

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

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<archive_shuffle_t<DATA_T>::value>::type startRecv(
      const GRAPH_T& frag,
      typename GRAPH_T::template vertex_array_t<DATA_T>& data, int thread_num) {
    std::vector<std::thread> threads(thread_num);
    std::atomic<fid_t> cur_fid(0);
    std::vector<size_t> out_archive_sizes(fnum_), in_archive_sizes(fnum_);
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        while (true) {
          fid_t got = cur_fid.fetch_add(1);
          if (got >= fnum_) {
            break;
          }
          auto& arc = shuffle_out_archives_[got];
          arc.Clear();
          auto& v_set = frag.MirrorVertices(got);
          for (auto v : v_set) {
            arc << data[v];
          }
          out_archive_sizes[comm_spec_.FragToWorker(got)] = arc.GetSize();
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }

    sync_comm::AllToAll(out_archive_sizes, in_archive_sizes, comm_spec_.comm());

    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto& buffer = shuffle_in_buffers_[src_fid];
      buffer.resize(in_archive_sizes[src_fid]);
      int old_req_num = recv_reqs_.size();
      sync_comm::irecv_buffer<char>(buffer.data(), buffer.size(),
                                    comm_spec_.FragToWorker(src_fid), 0, comm_,
                                    recv_reqs_);
      int new_req_num = recv_reqs_.size();
      recv_from_.resize(new_req_num, src_fid);
      remaining_reqs_[src_fid] = new_req_num - old_req_num;
    }

    post_process_handle_ = std::make_shared<
        batch_shuffle_message_manager_impl::PostProcess<GRAPH_T, DATA_T>>(
        frag, data, shuffle_in_buffers_);
  }

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<shuffle_inplace_t<GRAPH_T, DATA_T>::value>::type
  startRecv(const GRAPH_T& frag,
            typename GRAPH_T::template vertex_array_t<DATA_T>& data,
            int thread_num) {
    std::vector<int> req_offsets(fnum_);
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto range = frag.OuterVertices(src_fid);
      int old_req_num = recv_reqs_.size();
      int new_req_num =
          sync_comm::chunk_num<char>(range.size() * sizeof(DATA_T)) +
          old_req_num;
      recv_reqs_.resize(new_req_num);
      req_offsets[src_fid] = old_req_num;
      recv_from_.resize(new_req_num, src_fid);
      remaining_reqs_[src_fid] = new_req_num - old_req_num;
    }
    int fnum = static_cast<int>(fnum_);
    thread_num = (fnum - 1) < thread_num ? (fnum - 1) : thread_num;
    std::vector<std::thread> threads(thread_num);
    std::atomic<fid_t> cur(1);
    for (int i = 0; i < thread_num; ++i) {
      threads[i] = std::thread([&]() {
        while (true) {
          fid_t got = cur.fetch_add(1);
          if (got >= fnum_) {
            break;
          }
          fid_t src_fid = (fid_ + fnum_ - got) % fnum_;
          auto range = frag.OuterVertices(src_fid);
          sync_comm::irecv_buffer<char>(
              reinterpret_cast<char*>(&data[*range.begin()]),
              range.size() * sizeof(DATA_T), comm_spec_.FragToWorker(src_fid),
              0, comm_, &recv_reqs_[req_offsets[src_fid]]);
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }
  }

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<pod_shuffle_t<GRAPH_T, DATA_T>::value>::type
  startRecv(const GRAPH_T& frag,
            typename GRAPH_T::template vertex_array_t<DATA_T>& data,
            int thread_num) {
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto& buffer = shuffle_in_buffers_[src_fid];
      buffer.resize(frag.OuterVertices(src_fid).size() * sizeof(DATA_T));
      int old_req_num = recv_reqs_.size();
      sync_comm::irecv_buffer<char>(buffer.data(), buffer.size(),
                                    comm_spec_.FragToWorker(src_fid), 0, comm_,
                                    recv_reqs_);
      int new_req_num = recv_reqs_.size();
      recv_from_.resize(new_req_num, src_fid);
      remaining_reqs_[src_fid] = new_req_num - old_req_num;
    }
    post_process_handle_ = std::make_shared<
        batch_shuffle_message_manager_impl::PostProcess<GRAPH_T, DATA_T>>(
        frag, data, shuffle_in_buffers_);
  }

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<!archive_shuffle_t<DATA_T>::value>::type startSend(
      const GRAPH_T& frag,
      const typename GRAPH_T::template vertex_array_t<DATA_T>& data,
      int thread_num) {
    CHECK_EQ(sending_queue_.Size(), 0);
    int fnum = static_cast<int>(fnum_);
    sending_queue_.SetProducerNum(thread_num < fnum ? thread_num : 1);
    std::thread send_thread = std::thread([this]() {
      fid_t got;
      while (sending_queue_.Get(got)) {
        auto& vec = shuffle_out_buffers_[got];
        sync_comm::isend_buffer<char>(vec.data(), vec.size(),
                                      comm_spec_.FragToWorker(got), 0, comm_,
                                      send_reqs_);
      }
    });
    if (thread_num < fnum) {
      std::atomic<fid_t> cur(1);
      std::vector<std::thread> work_threads(thread_num);
      for (int i = 0; i < thread_num; ++i) {
        work_threads[i] = std::thread([&]() {
          while (true) {
            fid_t got = cur.fetch_add(1);
            if (got >= fnum_) {
              break;
            }
            fid_t dst_fid = (got + fid_) % fnum_;
            auto& id_vec = frag.MirrorVertices(dst_fid);
            auto& vec = shuffle_out_buffers_[dst_fid];
            vec.clear();
            vec.resize(id_vec.size() * sizeof(DATA_T));
            DATA_T* buf = reinterpret_cast<DATA_T*>(vec.data());
            size_t num = id_vec.size();
            for (size_t k = 0; k < num; ++k) {
              buf[k] = data[id_vec[k]];
            }
            sending_queue_.Put(dst_fid);
          }
          sending_queue_.DecProducerNum();
        });
      }
      for (auto& thrd : work_threads) {
        thrd.join();
      }
    } else {
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

        sending_queue_.Put(dst_fid);
        msg_size_ += vec.size();
      }
      sending_queue_.DecProducerNum();
    }
    send_thread.join();
  }

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<archive_shuffle_t<DATA_T>::value>::type startSend(
      const GRAPH_T& frag,
      const typename GRAPH_T::template vertex_array_t<DATA_T>& data,
      int thread_num) {
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (i + fid_) % fnum_;
      auto& arc = shuffle_out_archives_[dst_fid];
      sync_comm::isend_buffer<char>(arc.GetBuffer(), arc.GetSize(),
                                    comm_spec_.FragToWorker(dst_fid), 0, comm_,
                                    send_reqs_);
      msg_size_ += arc.GetSize();
    }
  }

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<!shuffle_inplace_t<GRAPH_T, DATA_T>::value>::type
  postProcess(const GRAPH_T& frag, fid_t i,
              const typename GRAPH_T::template vertex_array_t<DATA_T>& data,
              int thread_num) {
    if (i == fid_) {
      return;
    }
    auto& vec = shuffle_in_buffers_[i];
    OutArchive arc;
    arc.SetSlice(vec.data(), vec.size());
    auto& vertices = frag.OuterVertices(i);
    for (auto v : vertices) {
      arc >> data[v];
    }
  }

  template <typename GRAPH_T, typename DATA_T>
  typename std::enable_if<shuffle_inplace_t<GRAPH_T, DATA_T>::value>::type
  postProcess(const GRAPH_T& frag, fid_t i,
              const typename GRAPH_T::template vertex_array_t<DATA_T>& data,
              int thread_num) {}

  fid_t fid_;
  fid_t fnum_;
  CommSpec comm_spec_;

  MPI_Comm comm_;

  std::vector<std::vector<char>> shuffle_out_buffers_;
  std::vector<InArchive> shuffle_out_archives_;
  std::vector<std::vector<char>> shuffle_in_buffers_;

  std::shared_ptr<batch_shuffle_message_manager_impl::PostProcessBase>
      post_process_handle_;

  std::vector<MPI_Request> recv_reqs_;
  std::vector<fid_t> recv_from_;
  std::vector<int> remaining_reqs_;
  fid_t remaining_frags_;

  std::vector<MPI_Request> send_reqs_;

  size_t msg_size_;
  std::thread recv_thread_;

  bool to_terminate_;

  bool force_terminate_;
  TerminateInfo terminate_info_;

  BlockingQueue<fid_t> sending_queue_;
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_BATCH_SHUFFLE_MESSAGE_MANAGER_H_
