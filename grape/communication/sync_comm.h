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

#ifndef GRAPE_COMMUNICATION_SYNC_COMM_H_
#define GRAPE_COMMUNICATION_SYNC_COMM_H_

#include <glog/logging.h>
#include <mpi.h>

#include <limits>
#include <string>
#include <vector>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape {

#ifdef OPEN_MPI
#define NULL_COMM NULL
#else
#define NULL_COMM -1
#endif
#define ValidComm(comm) ((comm) != NULL_COMM)

inline void InitMPIComm() {
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
}

inline void FinalizeMPIComm() { MPI_Finalize(); }

static const int chunk_size = 409600;

template <typename T>
static inline void send_buffer(const T* ptr, size_t len, int dst_worker_id,
                               MPI_Comm comm, int tag) {
  const size_t chunk_size_in_bytes = chunk_size * sizeof(T);
  int iter = len / chunk_size;
  size_t remaining = (len % chunk_size) * sizeof(T);
  for (int i = 0; i < iter; ++i) {
    MPI_Send(ptr, chunk_size_in_bytes, MPI_CHAR, dst_worker_id, tag, comm);
    ptr += chunk_size;
  }
  if (remaining != 0) {
    MPI_Send(ptr, remaining, MPI_CHAR, dst_worker_id, tag, comm);
  }
}

template <typename T>
static inline void recv_buffer(T* ptr, size_t len, int src_worker_id,
                               MPI_Comm comm, int tag) {
  const size_t chunk_size_in_bytes = chunk_size * sizeof(T);
  int iter = len / chunk_size;
  size_t remaining = (len % chunk_size) * sizeof(T);
  for (int i = 0; i < iter; ++i) {
    MPI_Recv(ptr, chunk_size_in_bytes, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
    ptr += chunk_size;
  }
  if (remaining != 0) {
    MPI_Recv(ptr, remaining, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
  }
}

template <typename T>
inline void SendVector(const std::vector<T>& vec, int dst_worker_id,
                       MPI_Comm comm, int tag = 0) {
  size_t len = vec.size();
  MPI_Send(&len, sizeof(size_t), MPI_CHAR, dst_worker_id, tag, comm);
  send_buffer<T>(&vec[0], len, dst_worker_id, comm, tag);
}

template <typename T>
inline void RecvVector(std::vector<T>& vec, int src_worker_id, MPI_Comm comm,
                       int tag = 0) {
  size_t len;
  MPI_Recv(&len, sizeof(size_t), MPI_CHAR, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
  vec.resize(len);
  recv_buffer<T>(&vec[0], len, src_worker_id, comm, tag);
}

inline void SendArchive(const InArchive& archive, int dst_worker_id,
                        MPI_Comm comm, int tag = 0) {
  size_t len = archive.GetSize();
  MPI_Send(&len, sizeof(size_t), MPI_CHAR, dst_worker_id, tag, comm);
  int iter = len / chunk_size;
  int remaining = len % chunk_size;
  const char* ptr = archive.GetBuffer();
  for (int i = 0; i < iter; ++i) {
    MPI_Send(ptr, chunk_size, MPI_CHAR, dst_worker_id, tag, comm);
    ptr += chunk_size;
  }
  if (remaining != 0) {
    MPI_Send(ptr, remaining, MPI_CHAR, dst_worker_id, tag, comm);
  }
}

inline void RecvArchive(OutArchive& archive, int src_worker_id, MPI_Comm comm,
                        int tag = 0) {
  size_t len;
  MPI_Recv(&len, sizeof(size_t), MPI_CHAR, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
  archive.Clear();
  archive.Allocate(len);
  int iter = len / chunk_size;
  int remaining = len % chunk_size;
  char* ptr = archive.GetBuffer();
  for (int i = 0; i < iter; ++i) {
    MPI_Recv(ptr, chunk_size, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
    ptr += chunk_size;
  }
  if (remaining != 0) {
    MPI_Recv(ptr, remaining, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
  }
}

template <class T>
void BcastSend(const T& object, MPI_Comm comm) {
  InArchive ia;
  ia << object;
  size_t buf_size = ia.GetSize();
  int root;
  MPI_Comm_rank(comm, &root);
  MPI_Bcast(&buf_size, sizeof(size_t), MPI_CHAR, root, comm);
  CHECK_LT(buf_size, std::numeric_limits<int>::max());
  MPI_Bcast(ia.GetBuffer(), buf_size, MPI_CHAR, root, comm);
}

template <class T>
void BcastRecv(T& object, MPI_Comm comm, int root) {
  size_t buf_size;
  MPI_Bcast(&buf_size, sizeof(size_t), MPI_CHAR, root, comm);
  OutArchive oa(buf_size);
  CHECK_LT(buf_size, std::numeric_limits<int>::max());
  MPI_Bcast(oa.GetBuffer(), buf_size, MPI_CHAR, root, comm);
  oa >> object;
}

template <class T>
static void GlobalAllGatherv(T& object, std::vector<T>& to_exchange,
                             MPI_Comm comm, int worker_num) {
  grape::InArchive ia;
  ia << object;
  size_t send_count = ia.GetSize();
  auto* recv_counts = static_cast<int*>(malloc(sizeof(int) * worker_num));
  MPI_Allgather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

  size_t recv_buf_len = 0;
  for (int i = 0; i < worker_num; i++) {
    recv_buf_len += recv_counts[i];
  }

  grape::OutArchive oa(recv_buf_len);

  int* displs = static_cast<int*>(malloc(sizeof(size_t) * worker_num));
  displs[0] = 0;
  for (int i = 1; i < worker_num; i++) {
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }
  MPI_Allgatherv(ia.GetBuffer(), send_count, MPI_CHAR, oa.GetBuffer(),
                 recv_counts, displs, MPI_CHAR, comm);
  to_exchange.resize(worker_num);
  for (int i = 0; i < worker_num; i++) {
    oa >> to_exchange[i];
  }
  free(recv_counts);
  free(displs);
}
}  // namespace grape

#endif  // GRAPE_COMMUNICATION_SYNC_COMM_H_
