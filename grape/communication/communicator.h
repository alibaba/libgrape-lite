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

#ifndef GRAPE_COMMUNICATION_COMMUNICATOR_H_
#define GRAPE_COMMUNICATION_COMMUNICATOR_H_

#include <mpi.h>

#include <algorithm>
#include <memory>

#include "grape/communication/sync_comm.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape {

/**
 * @brief Communicator provides methods to implement distributed aggregation,
 * such as Min/Max/Sum.
 *
 */
class Communicator {
 public:
  Communicator() : comm_(NULL_COMM) {}
  virtual ~Communicator() {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }
  void InitCommunicator(MPI_Comm comm) { MPI_Comm_dup(comm, &comm_); }

  template <typename T>
  void SendTo(fid_t fid, const T& msg);

  template <typename T>
  void RecvFrom(fid_t fid, T& msg);

  template <typename T, typename FUNC_T>
  void AllReduce(const T& msg_in, T& msg_out, const FUNC_T& func) {
    int worker_id, worker_num;
    MPI_Comm_rank(comm_, &worker_id);
    MPI_Comm_size(comm_, &worker_num);
    if (worker_id == 0) {
      msg_out = msg_in;
      for (int src_worker = 1; src_worker < worker_num; ++src_worker) {
        T got_msg;
        RecvFrom<T>(src_worker, got_msg);
        func(msg_out, got_msg);
      }
      for (int dst_worker = 1; dst_worker < worker_num; ++dst_worker) {
        SendTo<T>(dst_worker, msg_out);
      }
    } else {
      SendTo<T>(0, msg_in);
      RecvFrom<T>(0, msg_out);
    }
  }

  template <typename T>
  void AllGather(const T& msg_in, std::vector<T>& msg_out) {
    int worker_id, worker_num;
    MPI_Comm_rank(comm_, &worker_id);
    MPI_Comm_size(comm_, &worker_num);

    msg_out.resize(worker_num);
    msg_out[worker_id] = msg_in;
    std::thread send_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker = (worker_id + 1) % worker_num;
        SendTo<T>(dst_worker, msg_in);
      }
    });
    std::thread recv_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int src_worker = (worker_id + worker_num - 1) % worker_num;
        RecvFrom<T>(src_worker, msg_out[src_worker]);
      }
    });
    recv_thread.join();
    send_thread.join();
  }

  template <typename T>
  void AllGather(T&& msg_in, std::vector<T>& msg_out) {
    int worker_id, worker_num;
    MPI_Comm_rank(comm_, &worker_id);
    MPI_Comm_size(comm_, &worker_num);

    msg_out.resize(worker_num);
    std::thread send_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker = (worker_id + i) % worker_num;
        SendTo<T>(dst_worker, msg_in);
      }
    });
    std::thread recv_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int src_worker = (worker_id + worker_num - i) % worker_num;
        RecvFrom<T>(src_worker, msg_out[src_worker]);
      }
    });
    recv_thread.join();
    send_thread.join();
    msg_out[worker_id] = std::move(msg_in);
  }

  template <typename T>
  void Max(const T& msg_in, T& msg_out) {
    AllReduce<T>(msg_in, msg_out,
                 [](T& lhs, const T& rhs) { lhs = std::max(lhs, rhs); });
  }

  template <typename T>
  void Min(const T& msg_in, T& msg_out) {
    AllReduce<T>(msg_in, msg_out,
                 [](T& lhs, const T& rhs) { lhs = std::min(lhs, rhs); });
  }

  template <typename T>
  void Sum(const T& msg_in, T& msg_out) {
    AllReduce<T>(msg_in, msg_out, [](T& lhs, const T& rhs) { lhs += rhs; });
  }

 private:
  MPI_Comm comm_;
};

template <typename T>
inline void Communicator::SendTo(fid_t fid, const T& msg) {
  int dst_worker = fid;
  InArchive arc;
  arc << msg;
  SendArchive(arc, dst_worker, comm_);
}

template <>
inline void Communicator::SendTo<InArchive>(fid_t fid, const InArchive& msg) {
  int dst_worker = fid;
  size_t msg_count = msg.GetSize();
  InArchive arc;
  arc << msg_count;
  SendArchive(arc, dst_worker, comm_);
  send_buffer<char>(msg.GetBuffer(), msg_count, dst_worker, comm_, 0);
}

template <typename T>
inline void Communicator::RecvFrom(fid_t fid, T& msg) {
  int src_worker = fid;
  OutArchive arc;
  RecvArchive(arc, src_worker, comm_);
  arc >> msg;
}

template <>
inline void Communicator::RecvFrom<InArchive>(fid_t fid, InArchive& msg) {
  int src_worker = fid;
  size_t msg_count = 0;
  OutArchive arc;
  RecvArchive(arc, src_worker, comm_);
  arc >> msg_count;
  msg.Resize(msg_count);
  recv_buffer<char>(msg.GetBuffer(), msg_count, src_worker, comm_, 0);
}

template <typename APP_T>
typename std::enable_if<std::is_base_of<Communicator, APP_T>::value>::type
InitCommunicator(std::shared_ptr<APP_T> app, MPI_Comm comm) {
  app->InitCommunicator(comm);
}

template <typename APP_T>
typename std::enable_if<!std::is_base_of<Communicator, APP_T>::value>::type
InitCommunicator(std::shared_ptr<APP_T> app, MPI_Comm comm) {}

}  // namespace grape

#endif  // GRAPE_COMMUNICATION_COMMUNICATOR_H_
