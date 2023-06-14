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
#include <type_traits>
#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/types.h"
#include "grape/utils/string_view_vector.h"

namespace grape {

template <typename T, T... Ints>
struct integer_sequence {
  typedef T value_type;
  static constexpr std::size_t size() { return sizeof...(Ints); }
};

template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;

template <typename T, std::size_t N, T... Is>
struct make_integer_sequence : make_integer_sequence<T, N - 1, N - 1, Is...> {};

template <typename T, T... Is>
struct make_integer_sequence<T, 0, Is...> : integer_sequence<T, Is...> {};

template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

template <typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

#define DEFAULT_CHUNK_SIZE 4096

template <typename T>
struct ShuffleBuffer {
  using type = std::vector<T>;

  static void SendTo(const type& buffer, int dst_worker_id, int tag,
                     MPI_Comm comm) {
    sync_comm::Send(buffer, dst_worker_id, tag, comm);
  }

  static void RecvFrom(type& buffer, int src_worker_id, int tag,
                       MPI_Comm comm) {
    sync_comm::RecvAt<T>(buffer, buffer.size(), src_worker_id, tag, comm);
  }
};

template <>
struct ShuffleBuffer<nonstd::string_view> {
  using type = StringViewVector;

  static void SendTo(const type& buffer, int dst_worker_id, int tag,
                     MPI_Comm comm) {
    sync_comm::Send(buffer, dst_worker_id, tag, comm);
  }

  static void RecvFrom(type& buffer, int src_worker_id, int tag,
                       MPI_Comm comm) {
    if (buffer.size() == 0) {
      sync_comm::Recv(buffer, src_worker_id, tag, comm);
    } else {
      type delta;
      sync_comm::Recv(delta, src_worker_id, tag, comm);
      size_t num = delta.size();
      for (size_t i = 0; i < num; ++i) {
        buffer.push_back(delta[i]);
      }
    }
  }
};

template <typename First, typename... Rest>
struct ShuffleBufferTuple : public ShuffleBufferTuple<Rest...> {
  ShuffleBufferTuple() : ShuffleBufferTuple<Rest...>() {}
  ShuffleBufferTuple(const ShuffleBufferTuple& rhs)
      : ShuffleBufferTuple<Rest...>(rhs), first(rhs.first) {}
  ShuffleBufferTuple(ShuffleBufferTuple&& rhs)
      : ShuffleBufferTuple<Rest...>(std::move(rhs)),
        first(std::move(rhs.first)) {}
  ShuffleBufferTuple(const typename ShuffleBuffer<First>::type& b0,
                     const typename ShuffleBuffer<Rest>::type&... bx)
      : first(b0), ShuffleBufferTuple<Rest...>(bx...) {}
  ShuffleBufferTuple(typename ShuffleBuffer<First>::type&& b0,
                     typename ShuffleBuffer<Rest>::type&&... bx)
      : first(std::move(b0)), ShuffleBufferTuple<Rest...>(std::move(bx)...) {}

  static constexpr size_t tuple_size =
      ShuffleBufferTuple<Rest...>::tuple_size + 1;

  void Emplace(const First& v0, const Rest&... vx) {
    first.emplace_back(v0);
    ShuffleBufferTuple<Rest...>::Emplace(vx...);
  }

  void SetBuffers(const typename ShuffleBuffer<First>::type b0,
                  const typename ShuffleBuffer<Rest>::type&... bx) {
    first = b0;
    ShuffleBufferTuple<Rest...>::SetBuffers(bx...);
  }

  void SetBuffers(typename ShuffleBuffer<First>::type&& b0,
                  typename ShuffleBuffer<Rest>::type&&... bx) {
    first = std::move(b0);
    ShuffleBufferTuple<Rest...>::SetBuffers(std::move(bx)...);
  }

  void AppendBuffers(const typename ShuffleBuffer<First>::type& b0,
                     const typename ShuffleBuffer<Rest>::type&... bx) {
    for (auto& v : b0) {
      first.emplace_back(v);
    }
    ShuffleBufferTuple<Rest...>::AppendBuffers(bx...);
  }

  void AppendBuffers(typename ShuffleBuffer<First>::type&& b0,
                     typename ShuffleBuffer<Rest>::type&&... bx) {
    for (auto& v : b0) {
      first.emplace_back(std::move(v));
    }
    ShuffleBufferTuple<Rest...>::AppendBuffers(std::move(bx)...);
  }

  size_t size() const {
    size_t ret = ShuffleBufferTuple<Rest...>::size();
    return std::min(ret, first.size());
  }

  void resize(size_t size) {
    first.resize(size);
    ShuffleBufferTuple<Rest...>::resize(size);
  }

  void SendTo(int dst_worker_id, int tag, MPI_Comm comm) {
    ShuffleBuffer<First>::SendTo(first, dst_worker_id, tag, comm);
    ShuffleBufferTuple<Rest...>::SendTo(dst_worker_id, tag, comm);
  }

  void RecvFrom(int src_worker_id, int tag, MPI_Comm comm) {
    ShuffleBuffer<First>::RecvFrom(first, src_worker_id, tag, comm);
    ShuffleBufferTuple<Rest...>::RecvFrom(src_worker_id, tag, comm);
  }

  void Clear() {
    first.clear();
    ShuffleBufferTuple<Rest...>::Clear();
  }

  typename ShuffleBuffer<First>::type first;
};

template <typename First>
struct ShuffleBufferTuple<First> {
  ShuffleBufferTuple() {}
  ShuffleBufferTuple(const ShuffleBufferTuple& rhs) : first(rhs.first) {}
  ShuffleBufferTuple(ShuffleBufferTuple&& rhs) : first(std::move(rhs.first)) {}
  explicit ShuffleBufferTuple(const typename ShuffleBuffer<First>::type& b0)
      : first(b0) {}
  explicit ShuffleBufferTuple(typename ShuffleBuffer<First>::type&& b0)
      : first(std::move(b0)) {}

  static constexpr size_t tuple_size = 1;

  void Emplace(const First& v0) { first.emplace_back(v0); }

  void SetBuffers(const typename ShuffleBuffer<First>::type& b0) { first = b0; }

  void SetBuffers(typename ShuffleBuffer<First>::type&& b0) {
    first = std::move(b0);
  }

  void AppendBuffers(const typename ShuffleBuffer<First>::type& b0) {
    for (auto& v : b0) {
      first.emplace_back(v);
    }
  }

  void AppendBuffers(typename ShuffleBuffer<First>::type&& b0) {
    for (auto& v : b0) {
      first.emplace_back(std::move(v));
    }
  }

  size_t size() const { return first.size(); }

  void resize(size_t size) { first.resize(size); }

  void SendTo(int dst_worker_id, int tag, MPI_Comm comm) {
    ShuffleBuffer<First>::SendTo(first, dst_worker_id, tag, comm);
  }

  void RecvFrom(int src_worker_id, int tag, MPI_Comm comm) {
    ShuffleBuffer<First>::RecvFrom(first, src_worker_id, tag, comm);
  }

  void Clear() { first.clear(); }

  typename ShuffleBuffer<First>::type first;
};

template <std::size_t index, typename T>
struct ShuffleBufferTuple_element;

template <std::size_t index, typename First, typename... Tail>
struct ShuffleBufferTuple_element<index, ShuffleBufferTuple<First, Tail...>>
    : ShuffleBufferTuple_element<index - 1, ShuffleBufferTuple<Tail...>> {};

template <typename First, typename... Rest>
struct ShuffleBufferTuple_element<0, ShuffleBufferTuple<First, Rest...>> {
  using buffer_type = typename ShuffleBuffer<First>::type;
};

template <typename T>
struct add_ref {
  using type = T&;
};

template <typename T>
struct add_ref<T&> {
  using type = T&;
};

template <typename T>
struct add_const_ref {
  using type = const T&;
};

template <typename T>
struct add_const_ref<T&> {
  using type = const T&;
};

template <typename T>
struct add_const_ref<const T&> {
  using type = const T&;
};

template <std::size_t index, typename First, typename... Rest>
struct get_buffer_helper {
  static typename add_ref<typename ShuffleBufferTuple_element<
      index, ShuffleBufferTuple<First, Rest...>>::buffer_type>::type
  value(ShuffleBufferTuple<First, Rest...>& bt) {
    return get_buffer_helper<index - 1, Rest...>::value(bt);
  }

  static typename add_const_ref<typename ShuffleBufferTuple_element<
      index, ShuffleBufferTuple<First, Rest...>>::buffer_type>::type
  const_value(const ShuffleBufferTuple<First, Rest...>& bt) {
    return get_buffer_helper<index - 1, Rest...>::const_value(bt);
  }
};

template <typename First, typename... Rest>
struct get_buffer_helper<0, First, Rest...> {
  static typename ShuffleBuffer<First>::type& value(
      ShuffleBufferTuple<First, Rest...>& bt) {
    return bt.first;
  }

  static const typename ShuffleBuffer<First>::type& const_value(
      const ShuffleBufferTuple<First, Rest...>& bt) {
    return bt.first;
  }
};

template <std::size_t index, typename First, typename... Rest>
typename add_ref<typename ShuffleBufferTuple_element<
    index, ShuffleBufferTuple<First, Rest...>>::buffer_type>::type
get_buffer(ShuffleBufferTuple<First, Rest...>& bt) {
  return get_buffer_helper<index, First, Rest...>::value(bt);
}

template <std::size_t index, typename First, typename... Rest>
typename add_const_ref<typename ShuffleBufferTuple_element<
    index, ShuffleBufferTuple<First, Rest...>>::buffer_type>::type
get_const_buffer(const ShuffleBufferTuple<First, Rest...>& bt) {
  return get_buffer_helper<index, First, Rest...>::const_value(bt);
}

template <typename Tuple, typename Func, std::size_t... index>
void foreach_helper(const Tuple& t, const Func& func,
                    index_sequence<index...>) {
  size_t size = t.size();
  for (size_t i = 0; i < size; ++i) {
    func(get_const_buffer<index>(t)[i]...);
  }
}

template <typename Tuple, typename Func, std::size_t... index>
void foreach_rval_helper(Tuple& t, const Func& func, index_sequence<index...>) {
  size_t size = t.size();
  for (size_t i = 0; i < size; ++i) {
    func(std::move(get_buffer<index>(t)[i])...);
  }
}

template <typename Tuple, typename Func>
void foreach(Tuple& t, const Func& func) {
  foreach_helper(t, func, make_index_sequence<Tuple::tuple_size>{});
}

template <typename Tuple, typename Func>
void foreach_rval(Tuple& t, const Func& func) {
  foreach_rval_helper(t, func, make_index_sequence<Tuple::tuple_size>{});
}

struct frag_shuffle_header {
  frag_shuffle_header() = default;
  frag_shuffle_header(size_t s, fid_t f) : size(s), fid(f) {}
  ~frag_shuffle_header() = default;
  size_t size;
  fid_t fid;
};

template <typename... TYPES>
class ShuffleOut {
 public:
  ShuffleOut() {}
  ~ShuffleOut() {}

  void Init(MPI_Comm comm, int tag = 0, size_t cs = 4096) {
    comm_ = comm;
    tag_ = tag;
    chunk_size_ = cs;
    current_size_ = 0;
    comm_disabled_ = false;
  }
  void DisableComm() { comm_disabled_ = true; }
  void SetDestination(int dst_worker_id, fid_t dst_frag_id) {
    dst_worker_id_ = dst_worker_id;
    dst_frag_id_ = dst_frag_id;
  }

  void Clear() {
    current_size_ = 0;
    buffers_.Clear();
  }

  void Emplace(const TYPES&... rest) {
    buffers_.Emplace(rest...);
    ++current_size_;
    if (comm_disabled_) {
      return;
    }
    if (current_size_ >= chunk_size_) {
      issue();
      Clear();
    }
  }

  void AppendBuffers(const typename ShuffleBuffer<TYPES>::type&... bx) {
    if (current_size_ == 0) {
      buffers_.SetBuffers(bx...);
    } else {
      buffers_.AppendBuffers(bx...);
    }
    current_size_ = buffers_.size();
    buffers_.resize(current_size_);
    if (comm_disabled_) {
      return;
    }
    if (current_size_ >= chunk_size_) {
      issue();
      Clear();
    }
  }

  void AppendBuffers(typename ShuffleBuffer<TYPES>::type&&... bx) {
    if (current_size_ == 0) {
      buffers_.SetBuffer(std::move(bx)...);
    } else {
      buffers_.AppendBuffers(std::move(bx)...);
    }
    current_size_ = buffers_.size();
    buffers_.resize(current_size_);
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

  ShuffleBufferTuple<TYPES...>& buffers() { return buffers_; }
  const ShuffleBufferTuple<TYPES...>& buffers() const { return buffers_; }

 private:
  void issue() {
    frag_shuffle_header header(current_size_, dst_frag_id_);
    sync_comm::Send<frag_shuffle_header>(header, dst_worker_id_, tag_, comm_);

    if (current_size_) {
      buffers_.SendTo(dst_worker_id_, tag_, comm_);
    }
  }

  ShuffleBufferTuple<TYPES...> buffers_;
  size_t chunk_size_;
  size_t current_size_;
  int dst_worker_id_;
  fid_t dst_frag_id_;
  int tag_;
  bool comm_disabled_;

  MPI_Comm comm_;
};

template <typename... TYPES>
class ShuffleIn {
 public:
  ShuffleIn() {}
  ~ShuffleIn() {}

  void Init(fid_t fnum, MPI_Comm comm, int tag = 0) {
    remaining_frag_num_ = fnum - 1;
    comm_ = comm;
    tag_ = tag;
    current_size_ = 0;
  }

  int Recv(fid_t& fid) {
    MPI_Status status;
    frag_shuffle_header header;
    while (true) {
      if (remaining_frag_num_ == 0) {
        return -1;
      }
      MPI_Probe(MPI_ANY_SOURCE, tag_, comm_, &status);
      sync_comm::Recv<frag_shuffle_header>(header, status.MPI_SOURCE, tag_,
                                           comm_);
      if (header.size == 0) {
        --remaining_frag_num_;
      } else {
        int src_worker_id = status.MPI_SOURCE;
        current_size_ += header.size;
        fid = header.fid;
        buffers_.RecvFrom(src_worker_id, tag_, comm_);
        return src_worker_id;
      }
    }
  }

  int RecvFrom(int src_worker_id) {
    frag_shuffle_header header;
    sync_comm::Recv<frag_shuffle_header>(header, src_worker_id, tag_, comm_);
    if (header.size == 0) {
      --remaining_frag_num_;
    } else {
      current_size_ += header.size;
      buffers_.RecvFrom(src_worker_id, tag_, comm_);
    }
    return header.size;
  }

  bool Finished() { return remaining_frag_num_ == 0; }

  void Clear() {
    buffers_.Clear();
    current_size_ = 0;
  }

  size_t Size() const { return current_size_; }

  ShuffleBufferTuple<TYPES...>& buffers() { return buffers_; }
  const ShuffleBufferTuple<TYPES...>& buffers() const { return buffers_; }

 private:
  ShuffleBufferTuple<TYPES...> buffers_;
  fid_t remaining_frag_num_;
  int tag_;
  size_t current_size_;
  MPI_Comm comm_;
};

}  // namespace grape

#endif  // GRAPE_COMMUNICATION_SHUFFLE_H_
