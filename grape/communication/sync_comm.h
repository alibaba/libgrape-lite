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

#include <assert.h>
#include <glog/logging.h>
#include <mpi.h>

#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/utils/string_view_vector.h"

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

namespace sync_comm {

static constexpr int chunk_size = 536870912;

template <typename T>
static inline void send_small_buffer(const T* ptr, size_t len,
                                     int dst_worker_id, int tag,
                                     MPI_Comm comm) {
  size_t len_in_bytes = len * sizeof(T);
  assert(len_in_bytes <= chunk_size);
  MPI_Send(ptr, len_in_bytes, MPI_CHAR, dst_worker_id, tag, comm);
}

template <typename T>
static inline void isend_small_buffer(const T* ptr, size_t len,
                                      int dst_worker_id, int tag, MPI_Comm comm,
                                      MPI_Request& req) {
  size_t len_in_bytes = len * sizeof(T);
  assert(len_in_bytes <= chunk_size);
  MPI_Isend(ptr, len_in_bytes, MPI_CHAR, dst_worker_id, tag, comm, &req);
}

template <typename T>
static inline void recv_small_buffer(T* ptr, size_t len, int src_worker_id,
                                     int tag, MPI_Comm comm) {
  size_t len_in_bytes = len * sizeof(T);
  assert(len_in_bytes <= chunk_size);
  MPI_Recv(ptr, len_in_bytes, MPI_CHAR, src_worker_id, tag, comm,
           MPI_STATUS_IGNORE);
}

template <typename T>
static inline void irecv_small_buffer(T* ptr, size_t len, int src_worker_id,
                                      int tag, MPI_Comm comm,
                                      MPI_Request& req) {
  size_t len_in_bytes = len * sizeof(T);
  assert(len_in_bytes <= chunk_size);
  MPI_Irecv(ptr, len_in_bytes, MPI_CHAR, src_worker_id, tag, comm, &req);
}

template <typename T>
static inline void send_buffer(const T* ptr, size_t len, int dst_worker_id,
                               int tag, MPI_Comm comm) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  if (len <= chunk_num) {
    send_small_buffer(ptr, len, dst_worker_id, tag, comm);
    return;
  }
  const size_t chunk_size_in_bytes = chunk_num * sizeof(T);
  int iter = len / chunk_num;
  size_t remaining = (len % chunk_num) * sizeof(T);
  LOG(INFO) << "sending large buffer in " << iter + (remaining != 0)
            << " iterations";
  for (int i = 0; i < iter; ++i) {
    MPI_Send(ptr, chunk_size_in_bytes, MPI_CHAR, dst_worker_id, tag, comm);
    ptr += chunk_num;
  }
  if (remaining != 0) {
    MPI_Send(ptr, remaining, MPI_CHAR, dst_worker_id, tag, comm);
  }
}

template <typename T>
static inline void isend_buffer(const T* ptr, size_t len, int dst_worker_id,
                                int tag, MPI_Comm comm,
                                std::vector<MPI_Request>& reqs) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  if (len <= chunk_num) {
    MPI_Request req;
    isend_small_buffer(ptr, len, dst_worker_id, tag, comm, req);
    reqs.push_back(req);
    return;
  }
  const size_t chunk_size_in_bytes = chunk_num * sizeof(T);
  int iter = len / chunk_num;
  size_t remaining = (len % chunk_num) * sizeof(T);
  LOG(INFO) << "isending large buffer in " << iter + (remaining != 0)
            << " iterations";
  for (int i = 0; i < iter; ++i) {
    MPI_Request req;
    MPI_Isend(ptr, chunk_size_in_bytes, MPI_CHAR, dst_worker_id, tag, comm,
              &req);
    reqs.push_back(req);
    ptr += chunk_num;
  }
  if (remaining != 0) {
    MPI_Request req;
    MPI_Isend(ptr, remaining, MPI_CHAR, dst_worker_id, tag, comm, &req);
    reqs.push_back(req);
  }
}

template <typename T>
static inline void recv_buffer(T* ptr, size_t len, int src_worker_id, int tag,
                               MPI_Comm comm) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  if (len <= chunk_num) {
    recv_small_buffer(ptr, len, src_worker_id, tag, comm);
    return;
  }
  const size_t chunk_size_in_bytes = chunk_num * sizeof(T);
  int iter = len / chunk_num;
  size_t remaining = (len % chunk_num) * sizeof(T);
  LOG(INFO) << "recving large buffer in " << iter + (remaining != 0)
            << " iterations";
  for (int i = 0; i < iter; ++i) {
    MPI_Recv(ptr, chunk_size_in_bytes, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
    ptr += chunk_num;
  }
  if (remaining != 0) {
    MPI_Recv(ptr, remaining, MPI_CHAR, src_worker_id, tag, comm,
             MPI_STATUS_IGNORE);
  }
}

template <typename T>
static inline int chunk_num(size_t len) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  return (len + chunk_num - 1) / chunk_num;
}

template <typename T>
static inline void irecv_buffer(T* ptr, size_t len, int src_worker_id, int tag,
                                MPI_Comm comm, MPI_Request* reqs) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  if (len <= chunk_num) {
    irecv_small_buffer(ptr, len, src_worker_id, tag, comm, reqs[0]);
    return;
  }
  const size_t chunk_size_in_bytes = chunk_num * sizeof(T);
  int iter = len / chunk_num;
  size_t remaining = (len % chunk_num) * sizeof(T);
  LOG(INFO) << "irecving large buffer in " << iter + (remaining != 0)
            << " iterations";
  for (int i = 0; i < iter; ++i) {
    MPI_Irecv(ptr, chunk_size_in_bytes, MPI_CHAR, src_worker_id, tag, comm,
              &reqs[i]);
    ptr += chunk_num;
  }
  if (remaining != 0) {
    MPI_Irecv(ptr, remaining, MPI_CHAR, src_worker_id, tag, comm, &reqs[iter]);
  }
}

template <typename T>
static inline void irecv_buffer(T* ptr, size_t len, int src_worker_id, int tag,
                                MPI_Comm comm, std::vector<MPI_Request>& reqs) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  if (len <= chunk_num) {
    MPI_Request req;
    irecv_small_buffer(ptr, len, src_worker_id, tag, comm, req);
    reqs.push_back(req);
    return;
  }
  const size_t chunk_size_in_bytes = chunk_num * sizeof(T);
  int iter = len / chunk_num;
  size_t remaining = (len % chunk_num) * sizeof(T);
  LOG(INFO) << "irecving large buffer in " << iter + (remaining != 0)
            << " iterations";
  for (int i = 0; i < iter; ++i) {
    MPI_Request req;
    MPI_Irecv(ptr, chunk_size_in_bytes, MPI_CHAR, src_worker_id, tag, comm,
              &req);
    reqs.push_back(req);
    ptr += chunk_num;
  }
  if (remaining != 0) {
    MPI_Request req;
    MPI_Irecv(ptr, remaining, MPI_CHAR, src_worker_id, tag, comm, &req);
    reqs.push_back(req);
  }
}

template <typename T>
static inline void bcast_small_buffer(T* ptr, size_t len, int root,
                                      MPI_Comm comm) {
  size_t len_in_bytes = len * sizeof(T);
  assert(len_in_bytes <= chunk_size);
  MPI_Bcast(ptr, len_in_bytes, MPI_CHAR, root, comm);
}

template <typename T>
static inline void bcast_buffer(T* ptr, size_t len, int root, MPI_Comm comm) {
  static constexpr size_t chunk_num = chunk_size / sizeof(T);
  if (len <= chunk_num) {
    bcast_small_buffer(ptr, len, root, comm);
    return;
  }
  const size_t chunk_size_in_bytes = chunk_num * sizeof(T);
  int iter = len / chunk_num;
  size_t remaining = (len % chunk_num) * sizeof(T);
  LOG(INFO) << "bcast large buffer in " << iter + (remaining != 0)
            << " iterations";
  for (int i = 0; i < iter; ++i) {
    MPI_Bcast(ptr, chunk_size_in_bytes, MPI_CHAR, root, comm);
    ptr += chunk_num;
  }
  if (remaining != 0) {
    MPI_Bcast(ptr, remaining, MPI_CHAR, root, comm);
  }
}

template <class T, class Enable = void>
struct CommImpl {
  static void send(const T& value, int dst_worker_id, int tag, MPI_Comm comm) {
    InArchive arc;
    arc << value;
    int64_t len = arc.GetSize();
    send_small_buffer<int64_t>(&len, 1, dst_worker_id, tag, comm);
    if (len > 0) {
      send_buffer<char>(arc.GetBuffer(), len, dst_worker_id, tag, comm);
    }
  }

  template <typename ITER_T>
  static void multiple_send(const T& value, const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    InArchive arc;
    arc << value;
    int64_t len = arc.GetSize();
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send_small_buffer<int64_t>(&len, 1, dst_worker_id, tag, comm);
      if (len > 0) {
        send_buffer<char>(arc.GetBuffer(), len, dst_worker_id, tag, comm);
      }
    }
  }

  static void recv(T& value, int src_worker_id, int tag, MPI_Comm comm) {
    int64_t len;
    recv_small_buffer<int64_t>(&len, 1, src_worker_id, tag, comm);
    if (len > 0) {
      OutArchive arc(len);
      recv_buffer<char>(arc.GetBuffer(), len, src_worker_id, tag, comm);
      arc >> value;
    }
  }

  static void bcast(T& value, int root, MPI_Comm comm) {
    int worker_id;
    MPI_Comm_rank(comm, &worker_id);
    if (worker_id == root) {
      InArchive arc;
      arc << value;
      int64_t len = arc.GetSize();
      bcast_small_buffer<int64_t>(&len, 1, root, comm);
      bcast_buffer<char>(arc.GetBuffer(), arc.GetSize(), root, comm);
    } else {
      int64_t len;
      bcast_small_buffer<int64_t>(&len, 1, root, comm);
      OutArchive arc(len);
      bcast_buffer<char>(arc.GetBuffer(), arc.GetSize(), root, comm);
      arc >> value;
    }
  }
};

template <class T>
struct CommImpl<T, typename std::enable_if<std::is_pod<T>::value>::type> {
  static void send(const T& value, int dst_worker_id, int tag, MPI_Comm comm) {
    send_small_buffer<T>(&value, 1, dst_worker_id, tag, comm);
  }

  static void recv(T& value, int src_worker_id, int tag, MPI_Comm comm) {
    recv_small_buffer<T>(&value, 1, src_worker_id, tag, comm);
  }

  template <typename ITER_T>
  static void multiple_send(const T& value, const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send(value, dst_worker_id, tag, comm);
    }
  }

  static void bcast(T& value, int root, MPI_Comm comm) {
    bcast_small_buffer<T>(&value, 1, root, comm);
  }
};

template <class T>
struct CommImpl<std::vector<T>,
                typename std::enable_if<std::is_pod<T>::value>::type> {
  static void send(const std::vector<T>& vec, int dst_worker_id, int tag,
                   MPI_Comm comm) {
    int64_t len = vec.size();
    CommImpl<int64_t>::send(len, dst_worker_id, tag, comm);
    if (len > 0) {
      send_buffer<T>(vec.data(), vec.size(), dst_worker_id, tag, comm);
    }
  }

  static void send_partial(const std::vector<T>& vec, size_t from, size_t to,
                           int dst_worker_id, int tag, MPI_Comm comm) {
    int64_t len = to - from;
    CommImpl<int64_t>::send(len, dst_worker_id, tag, comm);
    if (len > 0) {
      send_buffer<T>(vec.data() + from, len, dst_worker_id, tag, comm);
    }
  }

  static void recv(std::vector<T>& vec, int src_worker_id, int tag,
                   MPI_Comm comm) {
    int64_t len;
    CommImpl<int64_t>::recv(len, src_worker_id, tag, comm);
    vec.resize(len);
    if (len > 0) {
      recv_buffer<T>(vec.data(), vec.size(), src_worker_id, tag, comm);
    }
  }

  static void recv_at(std::vector<T>& vec, size_t offset, int src_worker_id,
                      int tag, MPI_Comm comm) {
    int64_t len;
    CommImpl<int64_t>::recv(len, src_worker_id, tag, comm);
    if (offset + len > vec.size()) {
      vec.resize(offset + len);
    }
    if (len > 0) {
      recv_buffer<T>(vec.data() + offset, len, src_worker_id, tag, comm);
    }
  }

  template <typename ITER_T>
  static void multiple_send(const std::vector<T>& vec,
                            const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send(vec, dst_worker_id, tag, comm);
    }
  }

  static void bcast(std::vector<T>& vec, int root, MPI_Comm comm) {
    int64_t len = vec.size();
    bcast_small_buffer<int64_t>(&len, 1, root, comm);
    vec.resize(len);
    bcast_buffer<T>(vec.data(), len, root, comm);
  }
};

template <>
struct CommImpl<InArchive, void> {
  static void send(const InArchive& arc, int dst_worker_id, int tag,
                   MPI_Comm comm) {
    int64_t len = arc.GetSize();
    CommImpl<int64_t>::send(len, dst_worker_id, tag, comm);
    if (len > 0) {
      send_buffer<char>(arc.GetBuffer(), arc.GetSize(), dst_worker_id, tag,
                        comm);
    }
  }

  static void recv(InArchive& arc, int src_worker_id, int tag, MPI_Comm comm) {
    int64_t len;
    CommImpl<int64_t>::recv(len, src_worker_id, tag, comm);
    arc.Resize(len);
    if (len > 0) {
      recv_buffer<char>(arc.GetBuffer(), len, src_worker_id, tag, comm);
    }
  }

  template <typename ITER_T>
  static void multiple_send(const InArchive& arc, const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send(arc, dst_worker_id, tag, comm);
    }
  }

  static void bcast(InArchive& arc, int root, MPI_Comm comm) {
    int64_t len = arc.GetSize();
    bcast_small_buffer<int64_t>(&len, 1, root, comm);
    arc.Resize(len);
    bcast_buffer<char>(arc.GetBuffer(), len, root, comm);
  }
};

template <>
struct CommImpl<OutArchive, void> {
  static void send(const OutArchive& arc, int dst_worker_id, int tag,
                   MPI_Comm comm) {
    int64_t len = arc.GetSize();
    CommImpl<int64_t>::send(len, dst_worker_id, tag, comm);
    if (len > 0) {
      send_buffer<char>(arc.GetBuffer(), arc.GetSize(), dst_worker_id, tag,
                        comm);
    }
  }

  static void recv(OutArchive& arc, int src_worker_id, int tag, MPI_Comm comm) {
    int64_t len;
    CommImpl<int64_t>::recv(len, src_worker_id, tag, comm);
    arc.Clear();
    if (len > 0) {
      arc.Allocate(len);
      recv_buffer<char>(arc.GetBuffer(), len, src_worker_id, tag, comm);
    }
  }

  template <typename ITER_T>
  static void multiple_send(const OutArchive& arc,
                            const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send(arc, dst_worker_id, tag, comm);
    }
  }

  static void bcast(OutArchive& arc, int root, MPI_Comm comm) {
    int worker_id;
    MPI_Comm_rank(comm, &worker_id);
    int64_t len = arc.GetSize();
    bcast_small_buffer<int64_t>(&len, 1, root, comm);
    if (root != worker_id) {
      arc.Clear();
      arc.Allocate(len);
    }
    bcast_buffer<char>(arc.GetBuffer(), len, root, comm);
  }
};

template <>
struct CommImpl<StringViewVector, void> {
  static void send(const StringViewVector& vec, int dst_worker_id, int tag,
                   MPI_Comm comm) {
    CommImpl<std::vector<char>>::send(vec.content_buffer(), dst_worker_id, tag,
                                      comm);
    CommImpl<std::vector<size_t>>::send(vec.offset_buffer(), dst_worker_id, tag,
                                        comm);
  }

  static void recv(StringViewVector& vec, int src_worker_id, int tag,
                   MPI_Comm comm) {
    CommImpl<std::vector<char>>::recv(vec.content_buffer(), src_worker_id, tag,
                                      comm);
    CommImpl<std::vector<size_t>>::recv(vec.offset_buffer(), src_worker_id, tag,
                                        comm);
  }

  template <typename ITER_T>
  static void multiple_send(const StringViewVector& vec,
                            const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send(vec, dst_worker_id, tag, comm);
    }
  }

  static void bcast(StringViewVector& vec, int root, MPI_Comm comm) {
    int worker_id;
    MPI_Comm_rank(comm, &worker_id);
    size_t len[2];
    if (worker_id == root) {
      len[0] = vec.content_buffer().size();
      len[1] = vec.offset_buffer().size();
    }
    bcast_small_buffer<size_t>(len, 2, root, comm);
    if (worker_id != root) {
      vec.content_buffer().resize(len[0]);
      vec.offset_buffer().resize(len[1]);
    }
    bcast_buffer<char>(vec.content_buffer().data(), len[0], root, comm);
    bcast_buffer<size_t>(vec.offset_buffer().data(), len[1], root, comm);
  }
};

template <class T>
struct CommImpl<std::vector<T>,
                typename std::enable_if<!std::is_pod<T>::value>::type> {
  static void send(const std::vector<T>& vec, int dst_worker_id, int tag,
                   MPI_Comm comm) {
    InArchive arc;
    arc << vec;
    CommImpl<InArchive>::send(arc, dst_worker_id, tag, comm);
  }

  static void send_partial(const std::vector<T>& vec, size_t from, size_t to,
                           int dst_worker_id, int tag, MPI_Comm comm) {
    InArchive arc;
    arc << (to - from);
    while (from != to) {
      arc << vec[from++];
    }
    CommImpl<InArchive>::send(arc, dst_worker_id, tag, comm);
  }

  static void recv(std::vector<T>& vec, int src_worker_id, int tag,
                   MPI_Comm comm) {
    OutArchive arc;
    CommImpl<OutArchive>::recv(arc, src_worker_id, tag, comm);
    arc >> vec;
  }

  static void recv_at(std::vector<T>& vec, size_t offset, int src_worker_id,
                      int tag, MPI_Comm comm) {
    OutArchive arc;
    CommImpl<OutArchive>::recv(arc, src_worker_id, tag, comm);
    size_t num;
    arc >> num;
    if (num + offset > vec.size()) {
      vec.resize(num + offset);
    }
    while (num--) {
      arc >> vec[offset++];
    }
  }

  template <typename ITER_T>
  static void multiple_send(const std::vector<T>& vec,
                            const ITER_T& worker_id_begin,
                            const ITER_T& worker_id_end, int tag,
                            MPI_Comm comm) {
    InArchive arc;
    arc << vec;
    int64_t len = arc.GetSize();
    for (ITER_T iter = worker_id_begin; iter != worker_id_end; ++iter) {
      int dst_worker_id = *iter;
      send_small_buffer<int64_t>(&len, 1, dst_worker_id, tag, comm);
      if (len > 0) {
        send_buffer<char>(arc.GetBuffer(), len, dst_worker_id, tag, comm);
      }
    }
  }

  static void bcast(std::vector<T>& vec, int root, MPI_Comm comm) {
    int worker_id;
    MPI_Comm_rank(comm, &worker_id);
    if (worker_id == root) {
      InArchive arc;
      arc << vec;
      int64_t len = arc.GetSize();
      bcast_small_buffer<int64_t>(&len, 1, root, comm);
      bcast_buffer<char>(arc.GetBuffer(), arc.GetSize(), root, comm);
    } else {
      int64_t len;
      bcast_small_buffer<int64_t>(&len, 1, root, comm);
      OutArchive arc(len);
      bcast_buffer<char>(arc.GetBuffer(), arc.GetSize(), root, comm);
      arc >> vec;
    }
  }
};

template <typename T>
void Send(const T& obj, int dst_worker_id, int tag, MPI_Comm comm) {
  CommImpl<T>::send(obj, dst_worker_id, tag, comm);
}

template <typename T>
void Recv(T& obj, int src_worker_id, int tag, MPI_Comm comm) {
  CommImpl<T>::recv(obj, src_worker_id, tag, comm);
}

template <typename T>
void SendPartial(const std::vector<T>& vec, size_t from, size_t to,
                 int dst_worker_id, int tag, MPI_Comm comm) {
  CommImpl<std::vector<T>>::send_partial(vec, from, to, dst_worker_id, tag,
                                         comm);
}

template <typename T>
void RecvAt(std::vector<T>& vec, size_t offset, int src_worker_id, int tag,
            MPI_Comm comm) {
  CommImpl<std::vector<T>>::recv_at(vec, offset, src_worker_id, tag, comm);
}

template <typename T>
void Bcast(T& object, int root, MPI_Comm comm) {
  CommImpl<T>::bcast(object, root, comm);
}

class WorkerIterator {
 public:
  WorkerIterator(int cur, int num) noexcept : cur_(cur), num_(num) {}
  ~WorkerIterator() = default;

  WorkerIterator& operator++() noexcept {
    cur_ = (cur_ + 1) % num_;
    return *this;
  }

  WorkerIterator operator++(int) noexcept {
    int prev = cur_;
    cur_ = (cur_ + 1) % num_;
    return WorkerIterator(prev, num_);
  }

  int operator*() const noexcept { return cur_; }

  bool operator==(const WorkerIterator& rhs) noexcept {
    return cur_ == rhs.cur_;
  }

  bool operator!=(const WorkerIterator& rhs) noexcept {
    return cur_ != rhs.cur_;
  }

 private:
  int cur_;
  int num_;
};

template <class T>
typename std::enable_if<std::is_pod<T>::value>::type AllGather(
    std::vector<T>& objects, MPI_Comm comm) {
  MPI_Allgather(MPI_IN_PLACE, sizeof(T), MPI_CHAR, objects.data(), sizeof(T),
                MPI_CHAR, comm);
}

template <class T>
typename std::enable_if<!std::is_pod<T>::value>::type AllGather(
    std::vector<T>& objects, MPI_Comm comm) {
  MPI_Barrier(comm);
  int worker_id, worker_num;
  MPI_Comm_rank(comm, &worker_id);
  MPI_Comm_size(comm, &worker_num);
  std::thread send_thread([&]() {
    CommImpl<T>::multiple_send(
        objects[worker_id],
        WorkerIterator((worker_id + 1) % worker_num, worker_num),
        WorkerIterator(worker_id, worker_num), 0, comm);
  });
  std::thread recv_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      CommImpl<T>::recv(objects[src_worker_id], src_worker_id, 0, comm);
    }
  });

  send_thread.join();
  recv_thread.join();
}

template <class T>
typename std::enable_if<std::is_pod<T>::value>::type AllToAll(
    const std::vector<T>& sendbuf, std::vector<T>& recvbuf, MPI_Comm comm) {
  int worker_id;
  MPI_Comm_rank(comm, &worker_id);
  MPI_Alltoall(sendbuf.data(), sizeof(T), MPI_CHAR, recvbuf.data(), sizeof(T),
               MPI_CHAR, comm);
  recvbuf[worker_id] = sendbuf[worker_id];
}

template <class T>
typename std::enable_if<!std::is_pod<T>::value>::type AllToAll(
    const std::vector<T>& sendbuf, std::vector<T>& recvbuf, MPI_Comm comm) {
  MPI_Barrier(comm);
  int worker_id, worker_num;
  MPI_Comm_rank(comm, &worker_id);
  MPI_Comm_size(comm, &worker_num);
  std::thread send_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int dst_worker_id = (worker_id + i) % worker_num;
      CommImpl<T>::send(sendbuf[dst_worker_id], dst_worker_id, 0, comm);
    }
  });
  std::thread recv_thread([&]() {
    recvbuf[worker_id] = sendbuf[worker_id];
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      CommImpl<T>::recv(recvbuf[src_worker_id], src_worker_id, 0, comm);
    }
  });

  send_thread.join();
  recv_thread.join();
}

template <typename T>
typename std::enable_if<std::is_pod<T>::value>::type FlatAllGather(
    const std::vector<T>& local, std::vector<T>& global, MPI_Comm comm) {
  int worker_id, worker_num;
  MPI_Comm_rank(comm, &worker_id);
  MPI_Comm_size(comm, &worker_num);
  std::vector<int64_t> sizes(worker_num);
  sizes[worker_id] = local.size();
  AllGather<int64_t>(sizes, comm);
  int64_t total_size = 0;
  for (auto size : sizes) {
    total_size += size;
  }
  global.resize(total_size);
  if (total_size * sizeof(T) <= chunk_size) {
    std::vector<int> counts(worker_num), displs(worker_num);
    int sum = 0;
    for (int i = 0; i < worker_num; ++i) {
      displs[i] = sum;
      counts[i] = sizes[i] * sizeof(T);
      sum += counts[i];
    }
    MPI_Allgatherv(local.data(), local.size() * sizeof(T), MPI_CHAR,
                   global.data(), counts.data(), displs.data(), MPI_CHAR, comm);
  } else {
    std::vector<MPI_Request> reqs;
    std::vector<int64_t> offsets;
    int64_t sum = 0;
    for (int i = 0; i < worker_num; ++i) {
      offsets[i] = sum;
      sum += sizes[i];
    }
    for (int i = 1; i < worker_num; ++i) {
      int dst_worker_id = (worker_id + i) % worker_num;
      isend_buffer<T>(local.data(), local.size(), dst_worker_id, 0, comm, reqs);
    }
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      irecv_buffer<T>(&global[offsets[src_worker_id]], sizes[src_worker_id],
                      src_worker_id, 0, comm, reqs);
    }
    memcpy(&global[offsets[worker_id]], local.data(),
           sizes[worker_id] * sizeof(T));
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
  }
}

template <typename T>
typename std::enable_if<!std::is_pod<T>::value>::type FlatAllGather(
    const std::vector<T>& local, std::vector<T>& global, MPI_Comm comm) {
  int worker_id, worker_num;
  MPI_Comm_rank(comm, &worker_id);
  MPI_Comm_size(comm, &worker_num);
  std::vector<int64_t> sizes(worker_num);
  sizes[worker_id] = local.size();
  AllGather<int64_t>(sizes, comm);
  int64_t total_size = 0;
  std::vector<int64_t> offsets;
  for (auto size : sizes) {
    offsets.push_back(total_size);
    total_size += size;
  }
  global.resize(total_size);
  std::thread send_thread([&]() {
    CommImpl<std::vector<T>>::multiple_send(
        local, WorkerIterator((worker_id + 1) % worker_num, worker_num),
        WorkerIterator(worker_id, worker_num), 0, comm);
    std::copy(local.begin(), local.end(), global.begin() + offsets[worker_id]);
  });
  std::thread recv_thread([&]() {
    for (int i = 1; i < worker_num; ++i) {
      int src_worker_id = (worker_id + worker_num - i) % worker_num;
      CommImpl<std::vector<T>>::recv_at(global, offsets[src_worker_id],
                                        src_worker_id, 0, comm);
    }
  });

  send_thread.join();
  recv_thread.join();
}

}  // namespace sync_comm

}  // namespace grape

#endif  // GRAPE_COMMUNICATION_SYNC_COMM_H_
