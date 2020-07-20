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

#ifndef GRAPE_COMMUNICATION_SHUFFLE_H_
#define GRAPE_COMMUNICATION_SHUFFLE_H_

#include <mpi.h>

#include <string>
#include <vector>

#include "grape/communication/sync_comm.h"

namespace grape {

#define DEFAULT_CHUNK_SIZE 4096

/**
 * @brief ShuffleUnit wraps a vector, for data shuffling between workers.
 *
 * @tparam T The data type to be shuffle.
 */
template <typename T>
class ShuffleUnit {
 public:
  ShuffleUnit() {}
  ~ShuffleUnit() {}

  using BufferT = std::vector<T>;
  using ValueT = T;

  void emplace(const ValueT& v) { buffer_.emplace_back(v); }
  void clear() { buffer_.clear(); }

  size_t size() const { return buffer_.size(); }

  BufferT& data() { return buffer_; }
  const BufferT& data() const { return buffer_; }

  void SendTo(int dst_worker_id, int tag, MPI_Comm comm) {
    size_t size = buffer_.size();
    MPI_Send(&size, static_cast<int>(sizeof(size_t)), MPI_CHAR, dst_worker_id,
             tag, comm);
    if (size) {
      MPI_Send(buffer_.data(), static_cast<int>(size * sizeof(T)), MPI_CHAR,
               dst_worker_id, tag, comm);
    }
  }

  void RecvFrom(int src_worker_id, int tag, MPI_Comm comm) {
    size_t old_size = buffer_.size();
    size_t to_recv;
    MPI_Recv(&to_recv, static_cast<int>(sizeof(size_t)), MPI_CHAR,
             src_worker_id, tag, comm, MPI_STATUS_IGNORE);
    if (to_recv) {
      buffer_.resize(to_recv + old_size);
      MPI_Recv(&buffer_[old_size], static_cast<int>(to_recv * sizeof(T)),
               MPI_CHAR, src_worker_id, tag, comm, MPI_STATUS_IGNORE);
    }
  }

 private:
  BufferT buffer_;
};

struct frag_shuffle_header {
  frag_shuffle_header() {}
  frag_shuffle_header(size_t s, fid_t f) : size(s), fid(f) {}
  size_t size;
  fid_t fid;
};

/**
 * @brief ShuffleOut for a <ShuffleUnit>.
 *
 * to exchange vertices when building fragments.
 *
 * @tparam T0, ValueT of the ShuffleUnit.
 */
template <typename T0>
class ShuffleOutUnary {
 public:
  ShuffleOutUnary()
      : current_size_(0),
        dst_worker_id_(-1),
        dst_frag_id_(0),
        tag_(0),
        comm_disabled_(false),
        comm_(NULL_COMM) {}
  ~ShuffleOutUnary() {}

  using BufferT0 = typename ShuffleUnit<T0>::BufferT;
  using ValueT0 = typename ShuffleUnit<T0>::ValueT;

  void Init(MPI_Comm comm, int tag = 0, size_t cs = DEFAULT_CHUNK_SIZE) {
    comm_ = comm;
    tag_ = tag;
    chunk_size_ = cs;
  }
  void DisableComm() { comm_disabled_ = true; }
  void SetDestination(int dst_worker_id, fid_t dst_frag_id) {
    dst_worker_id_ = dst_worker_id;
    dst_frag_id_ = dst_frag_id;
  }

  void Clear() {
    current_size_ = 0;
    su0.clear();
  }

  void Emplace(const ValueT0& v0) {
    su0.emplace(v0);
    ++current_size_;
    if (comm_disabled_) {
      return;
    }
    if (current_size_ >= chunk_size_) {
      issue();
      Clear();
    }
  }

  void Flush() {
    if (comm_disabled_) {
      return;
    }
    if (current_size_) {
      issue();
      Clear();
    }
    issue();
  }

  BufferT0& Buffer0() { return su0.data(); }
  const BufferT0& Buffer0() const { return su0.data(); }

 private:
  void issue() {
    frag_shuffle_header header(current_size_, dst_frag_id_);
    MPI_Send(&header, static_cast<int>(sizeof(frag_shuffle_header)), MPI_CHAR,
             dst_worker_id_, tag_, comm_);
    if (current_size_) {
      su0.SendTo(dst_worker_id_, tag_, comm_);
    }
  }

  ShuffleUnit<T0> su0;

  size_t chunk_size_ = 4096;
  size_t current_size_;
  int dst_worker_id_;
  fid_t dst_frag_id_;
  int tag_;
  bool comm_disabled_;

  MPI_Comm comm_;
};

/**
 * @brief ShuffleOut for two <ShuffleUnit>s.
 *
 * To exchange vertices and their attributes, or edges when building fragments.
 *
 * @tparam T0
 * @tparam T1
 */
template <typename T0, typename T1>
class ShuffleOutPair {
 public:
  ShuffleOutPair()
      : current_size_(0),
        dst_worker_id_(-1),
        dst_frag_id_(0),
        tag_(0),
        comm_disabled_(false),
        comm_(NULL_COMM) {}
  ~ShuffleOutPair() {}

  using BufferT0 = typename ShuffleUnit<T0>::BufferT;
  using BufferT1 = typename ShuffleUnit<T1>::BufferT;
  using ValueT0 = typename ShuffleUnit<T0>::ValueT;
  using ValueT1 = typename ShuffleUnit<T1>::ValueT;

  void Init(MPI_Comm comm, int tag = 0, size_t cs = DEFAULT_CHUNK_SIZE) {
    comm_ = comm;
    tag_ = tag;
    chunk_size_ = cs;
  }
  void DisableComm() { comm_disabled_ = true; }
  void SetDestination(int dst_worker_id, fid_t dst_frag_id) {
    dst_worker_id_ = dst_worker_id;
    dst_frag_id_ = dst_frag_id;
  }

  void Clear() {
    current_size_ = 0;
    su0.clear();
    su1.clear();
  }

  void Emplace(const ValueT0& v0, const ValueT1& v1) {
    su0.emplace(v0);
    su1.emplace(v1);
    ++current_size_;
    if (comm_disabled_) {
      return;
    }
    if (current_size_ >= chunk_size_) {
      issue();
      Clear();
    }
  }

  void Flush() {
    if (comm_disabled_) {
      return;
    }
    if (current_size_) {
      issue();
      Clear();
    }
    issue();
  }

  BufferT0& Buffer0() { return su0.data(); }
  const BufferT0& Buffer0() const { return su0.data(); }

  BufferT1& Buffer1() { return su1.data(); }
  const BufferT1& Buffer1() const { return su1.data(); }

 private:
  void issue() {
    frag_shuffle_header header(current_size_, dst_frag_id_);
    MPI_Send(&header, static_cast<int>(sizeof(frag_shuffle_header)), MPI_CHAR,
             dst_worker_id_, tag_, comm_);
    if (current_size_) {
      su0.SendTo(dst_worker_id_, tag_, comm_);
      su1.SendTo(dst_worker_id_, tag_, comm_);
    }
  }

  ShuffleUnit<T0> su0;
  ShuffleUnit<T1> su1;

  size_t chunk_size_ = 4096;
  size_t current_size_;
  int dst_worker_id_;
  fid_t dst_frag_id_;
  int tag_;
  bool comm_disabled_;

  MPI_Comm comm_;
};

/**
 * @brief ShuffleOut for three <ShuffleUnit>s.
 *
 * to exchange edges and their edges when building fragment.
 *
 * @tparam T0
 * @tparam T1
 * @tparam T2
 */
template <typename T0, typename T1, typename T2>
class ShuffleOutTriple {
 public:
  ShuffleOutTriple()
      : current_size_(0),
        dst_worker_id_(-1),
        dst_frag_id_(0),
        tag_(0),
        comm_disabled_(false),
        comm_(NULL_COMM) {}
  ~ShuffleOutTriple() {}

  using BufferT0 = typename ShuffleUnit<T0>::BufferT;
  using BufferT1 = typename ShuffleUnit<T1>::BufferT;
  using BufferT2 = typename ShuffleUnit<T2>::BufferT;
  using ValueT0 = typename ShuffleUnit<T0>::ValueT;
  using ValueT1 = typename ShuffleUnit<T1>::ValueT;
  using ValueT2 = typename ShuffleUnit<T2>::ValueT;

  void Init(MPI_Comm comm, int tag = 0, size_t cs = DEFAULT_CHUNK_SIZE) {
    comm_ = comm;
    tag_ = tag;
    chunk_size_ = cs;
  }
  void DisableComm() { comm_disabled_ = true; }
  void SetDestination(int dst_worker_id, fid_t dst_frag_id) {
    dst_worker_id_ = dst_worker_id;
    dst_frag_id_ = dst_frag_id;
  }

  void Clear() {
    current_size_ = 0;
    su0.clear();
    su1.clear();
    su2.clear();
  }

  void Emplace(const ValueT0& v0, const ValueT1& v1, const ValueT2& v2) {
    su0.emplace(v0);
    su1.emplace(v1);
    su2.emplace(v2);
    ++current_size_;
    if (comm_disabled_) {
      return;
    }
    if (current_size_ >= chunk_size_) {
      issue();
      Clear();
    }
  }

  void Flush() {
    if (comm_disabled_) {
      return;
    }
    if (current_size_) {
      issue();
      Clear();
    }
    issue();
  }

  BufferT0& Buffer0() { return su0.data(); }
  const BufferT0& Buffer0() const { return su0.data(); }

  BufferT1& Buffer1() { return su1.data(); }
  const BufferT1& Buffer1() const { return su1.data(); }

  BufferT2& Buffer2() { return su2.data(); }
  const BufferT2& Buffer2() const { return su2.data(); }

 private:
  void issue() {
    frag_shuffle_header header(current_size_, dst_frag_id_);
    MPI_Send(&header, static_cast<int>(sizeof(frag_shuffle_header)), MPI_CHAR,
             dst_worker_id_, tag_, comm_);
    if (current_size_) {
      su0.SendTo(dst_worker_id_, tag_, comm_);
      su1.SendTo(dst_worker_id_, tag_, comm_);
      su2.SendTo(dst_worker_id_, tag_, comm_);
    }
  }

  ShuffleUnit<T0> su0;
  ShuffleUnit<T1> su1;
  ShuffleUnit<T2> su2;

  size_t chunk_size_ = 4096;
  size_t current_size_;
  int dst_worker_id_;
  fid_t dst_frag_id_;
  int tag_;
  bool comm_disabled_;

  MPI_Comm comm_;
};

/**
 * @brief ShuffleIn for a <ShuffleUnit>.
 *
 * to exchange vertices when building fragments.
 *
 * @tparam T0, ValueT of the ShuffleUnit.
 */
template <typename T0>
class ShuffleInUnary {
 public:
  explicit ShuffleInUnary(fid_t frag_num)
      : remaining_frag_num_(frag_num), tag_(0), comm_(NULL_COMM) {}
  ~ShuffleInUnary() {}

  void Init(MPI_Comm comm, int tag = 0) {
    comm_ = comm;
    tag_ = tag;
  }

  int Recv(fid_t& fid) {
    MPI_Status status;
    frag_shuffle_header header;
    while (true) {
      if (remaining_frag_num_ == 0) {
        return -1;
      }
      MPI_Recv(&header, static_cast<int>(sizeof(frag_shuffle_header)), MPI_CHAR,
               MPI_ANY_SOURCE, tag_, comm_, &status);
      if (header.size == 0) {
        --remaining_frag_num_;
      } else {
        int src_worker_id = status.MPI_SOURCE;
        current_size_ = header.size;
        fid = header.fid;
        su0.clear();
        su0.RecvFrom(src_worker_id, tag_, comm_);
        return src_worker_id;
      }
    }
  }

  bool Finished() { return remaining_frag_num_ == 0; }

  void Clear() {
    su0.clear();
    current_size_ = 0;
  }

  size_t Size() const { return current_size_; }

  typename ShuffleUnit<T0>::BufferT& Buffer0() { return su0.data(); }
  const typename ShuffleUnit<T0>::BufferT& Buffer0() const {
    return su0.data();
  }

 private:
  fid_t remaining_frag_num_;
  int tag_;
  ShuffleUnit<T0> su0;
  size_t current_size_;

  MPI_Comm comm_;
};

/**
 * @brief ShuffleIn for two <ShuffleUnit>s.
 *
 * To exchange vertices and their attributes, or edges when building fragments.
 *
 * @tparam T0
 * @tparam T1
 */
template <typename T0, typename T1>
class ShuffleInPair {
 public:
  explicit ShuffleInPair(fid_t frag_num)
      : remaining_frag_num_(frag_num), tag_(0), comm_(NULL_COMM) {}
  ~ShuffleInPair() {}

  void Init(MPI_Comm comm, int tag = 0) {
    comm_ = comm;
    tag_ = tag;
  }

  int Recv(fid_t& fid) {
    MPI_Status status;
    frag_shuffle_header header;
    while (true) {
      if (remaining_frag_num_ == 0) {
        return -1;
      }
      MPI_Recv(&header, static_cast<int>(sizeof(frag_shuffle_header)), MPI_CHAR,
               MPI_ANY_SOURCE, tag_, comm_, &status);
      if (header.size == 0) {
        --remaining_frag_num_;
      } else {
        int src_worker_id = status.MPI_SOURCE;
        current_size_ = header.size;
        fid = header.fid;
        su0.clear();
        su1.clear();
        su0.RecvFrom(src_worker_id, tag_, comm_);
        su1.RecvFrom(src_worker_id, tag_, comm_);
        return src_worker_id;
      }
    }
  }

  bool Finished() { return remaining_frag_num_ == 0; }

  void Clear() {
    su0.clear();
    su1.clear();
    current_size_ = 0;
  }

  size_t Size() const { return current_size_; }

  typename ShuffleUnit<T0>::BufferT& Buffer0() { return su0.data(); }
  const typename ShuffleUnit<T0>::BufferT& Buffer0() const {
    return su0.data();
  }

  typename ShuffleUnit<T1>::BufferT& Buffer1() { return su1.data(); }
  const typename ShuffleUnit<T1>::BufferT& Buffer1() const {
    return su1.data();
  }

 private:
  fid_t remaining_frag_num_;
  int tag_;
  ShuffleUnit<T0> su0;
  ShuffleUnit<T1> su1;
  size_t current_size_;
  MPI_Comm comm_;
};

/**
 * @brief ShuffleIn for three <ShuffleUnit>s.
 *
 * to exchange edges and their edges when building fragment.
 *
 * @tparam T0
 * @tparam T1
 * @tparam T2
 */
template <typename T0, typename T1, typename T2>
class ShuffleInTriple {
 public:
  explicit ShuffleInTriple(fid_t frag_num)
      : remaining_frag_num_(frag_num), tag_(0), comm_(NULL_COMM) {}
  ~ShuffleInTriple() {}

  void Init(MPI_Comm comm, int tag = 0) {
    comm_ = comm;
    tag_ = tag;
  }

  int Recv(fid_t& fid) {
    MPI_Status status;
    frag_shuffle_header header;
    while (true) {
      if (remaining_frag_num_ == 0) {
        return -1;
      }
      MPI_Recv(&header, static_cast<int>(sizeof(frag_shuffle_header)), MPI_CHAR,
               MPI_ANY_SOURCE, tag_, comm_, &status);
      if (header.size == 0) {
        --remaining_frag_num_;
      } else {
        int src_worker_id = status.MPI_SOURCE;
        current_size_ = header.size;
        fid = header.fid;
        su0.clear();
        su1.clear();
        su2.clear();
        su0.RecvFrom(src_worker_id, tag_, comm_);
        su1.RecvFrom(src_worker_id, tag_, comm_);
        su2.RecvFrom(src_worker_id, tag_, comm_);
        return src_worker_id;
      }
    }
  }

  bool Finished() { return remaining_frag_num_ == 0; }

  void Clear() {
    su0.clear();
    su1.clear();
    su2.clear();
    current_size_ = 0;
  }

  size_t Size() const { return current_size_; }

  typename ShuffleUnit<T0>::BufferT& Buffer0() { return su0.data(); }
  const typename ShuffleUnit<T0>::BufferT& Buffer0() const {
    return su0.data();
  }

  typename ShuffleUnit<T1>::BufferT& Buffer1() { return su1.data(); }
  const typename ShuffleUnit<T1>::BufferT& Buffer1() const {
    return su1.data();
  }

  typename ShuffleUnit<T2>::BufferT& Buffer2() { return su2.data(); }
  const typename ShuffleUnit<T2>::BufferT& Buffer2() const {
    return su2.data();
  }

 private:
  fid_t remaining_frag_num_;
  int tag_;
  ShuffleUnit<T0> su0;
  ShuffleUnit<T1> su1;
  ShuffleUnit<T2> su2;
  size_t current_size_;

  MPI_Comm comm_;
};

#undef DEFAULT_CHUNK_SIZE

}  // namespace grape

#endif  // GRAPE_COMMUNICATION_SHUFFLE_H_
