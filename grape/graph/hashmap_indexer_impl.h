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

#ifndef GRAPE_GRAPH_HASHMAP_INDEXER_IMPL_H_
#define GRAPE_GRAPH_HASHMAP_INDEXER_IMPL_H_

#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/utils/ref_vector.h"
#include "grape/utils/string_view_vector.h"

namespace grape {

namespace hashmap_indexer_impl {

static constexpr int8_t min_lookups = 4;
static constexpr double max_load_factor = 0.5f;

template <typename T>
size_t vec_dump_bytes(T const& vec) {
  return vec.size() * sizeof(vec.front()) + sizeof(typename T::size_type);
}

template <typename T>
struct KeyBuffer {
 public:
  KeyBuffer() = default;
  ~KeyBuffer() = default;

  const T& get(size_t idx) const { return inner_[idx]; }
  void set(size_t idx, const T& val) { inner_[idx] = val; }

  void push_back(const T& val) { inner_.push_back(val); }

  size_t size() const { return inner_.size(); }

  std::vector<T, Allocator<T>>& buffer() { return inner_; }
  const std::vector<T, Allocator<T>>& buffer() const { return inner_; }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) const {
    size_t size = inner_.size();
    CHECK(writer->Write(&size, sizeof(size_t)));
    if (size > 0) {
      CHECK(writer->Write(const_cast<T*>(inner_.data()), size * sizeof(T)));
    }
  }

  void serialize_to_mem(std::vector<char>& buf) const {
    encode_vec(inner_, buf);
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    size_t size;
    CHECK(reader->Read(&size, sizeof(size_t)));
    if (size > 0) {
      inner_.resize(size);
      CHECK(reader->Read(inner_.data(), size * sizeof(T)));
    }
  }

  void swap(KeyBuffer& rhs) { inner_.swap(rhs.inner_); }

  void clear() { inner_.clear(); }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_vec(inner_);
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    dumper.dump_vec(inner_);
  }

  size_t dump_size() { return vec_dump_bytes(inner_); }

 private:
  std::vector<T, Allocator<T>> inner_;
};

template <>
struct KeyBuffer<nonstd::string_view> {
  KeyBuffer() = default;
  ~KeyBuffer() = default;

  nonstd::string_view get(size_t idx) const { return inner_[idx]; }

  void push_back(const nonstd::string_view& val) { inner_.push_back(val); }

  size_t size() const { return inner_.size(); }

  StringViewVector& buffer() { return inner_; }
  const StringViewVector& buffer() const { return inner_; }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) const {
    inner_.serialize(writer);
  }

  void serialize_to_mem(std::vector<char>& buf) const {
    inner_.serialize_to_mem(buf);
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    inner_.deserialize(reader);
  }

  void swap(KeyBuffer& rhs) { inner_.swap(rhs.inner_); }

  void clear() { inner_.clear(); }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_vec(inner_.content_buffer());
    loader.load_vec(inner_.offset_buffer());
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    dumper.dump_vec(inner_.content_buffer());
    dumper.dump_vec(inner_.offset_buffer());
  }

  size_t dump_size() {
    return vec_dump_bytes(inner_.content_buffer()) +
           vec_dump_bytes(inner_.offset_buffer());
  }

 private:
  StringViewVector inner_;
};

#if __cplusplus >= 201703L
template <>
struct KeyBuffer<std::string_view> {
  KeyBuffer() = default;
  ~KeyBuffer() = default;

  std::string_view get(size_t idx) const {
    std::string_view view(inner_[idx].data(), inner_[idx].size());
    return view;
  }

  void push_back(const std::string_view& val) {
    nonstd::string_view view(val.data(), val.size());
    inner_.push_back(view);
  }

  size_t size() const { return inner_.size(); }

  StringViewVector& buffer() { return inner_; }
  const StringViewVector& buffer() const { return inner_; }

  void swap(KeyBuffer& rhs) { inner_.swap(rhs.inner_); }

  void clear() { inner_.clear(); }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load_vec(inner_.content_buffer());
    loader.load_vec(inner_.offset_buffer());
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    dumper.dump_vec(inner_.content_buffer());
    dumper.dump_vec(inner_.offset_buffer());
  }

  size_t dump_size() {
    return vec_dump_bytes(inner_.content_buffer()) +
           vec_dump_bytes(inner_.offset_buffer());
  }

 private:
  StringViewVector inner_;
};
#endif

template <typename T>
struct KeyBufferView {
 public:
  KeyBufferView() {}

  size_t init(const void* buffer, size_t size) {
    return inner_.init(buffer, size);
  }

  T get(size_t idx) const { return inner_.get(idx); }

  size_t size() const { return inner_.size(); }

  template <typename Loader>
  void load(Loader& loader) {
    inner_.load(loader);
  }

 private:
  ref_vector<T> inner_;
};

}  // namespace hashmap_indexer_impl

namespace sync_comm {

template <typename T>
struct CommImpl<hashmap_indexer_impl::KeyBuffer<T>> {
  static void send(const hashmap_indexer_impl::KeyBuffer<T>& buf,
                   int dst_worker_id, int tag, MPI_Comm comm) {
    Send(buf.buffer(), dst_worker_id, tag, comm);
  }

  static void recv(hashmap_indexer_impl::KeyBuffer<T>& buf, int src_worker_id,
                   int tag, MPI_Comm comm) {
    Recv(buf.buffer(), src_worker_id, tag, comm);
  }
};

template <>
struct CommImpl<hashmap_indexer_impl::KeyBuffer<nonstd::string_view>> {
  static void send(
      const hashmap_indexer_impl::KeyBuffer<nonstd::string_view>& buf,
      int dst_worker_id, int tag, MPI_Comm comm) {
    Send(buf.buffer(), dst_worker_id, tag, comm);
  }

  static void recv(hashmap_indexer_impl::KeyBuffer<nonstd::string_view>& buf,
                   int src_worker_id, int tag, MPI_Comm comm) {
    Recv(buf.buffer(), src_worker_id, tag, comm);
  }
};

}  // namespace sync_comm

}  // namespace grape

#endif  // GRAPE_GRAPH_HASHMAP_INDEXER_IMPL_H_
