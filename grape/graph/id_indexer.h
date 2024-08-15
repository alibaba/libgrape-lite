
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

#ifndef GRAPE_GRAPH_ID_INDEXER_H_
#define GRAPE_GRAPH_ID_INDEXER_H_

#include <cmath>
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/communication/sync_comm.h"
#include "grape/config.h"
#include "grape/io/io_adaptor_base.h"
#include "grape/types.h"
#include "grape/utils/ref_vector.h"
#include "grape/utils/string_view_vector.h"

namespace grape {

namespace id_indexer_impl {

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

  size_t dump_size() const { return vec_dump_bytes(inner_); }

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

  size_t dump_size() const {
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

}  // namespace id_indexer_impl

namespace sync_comm {

template <typename T>
struct CommImpl<id_indexer_impl::KeyBuffer<T>> {
  static void send(const id_indexer_impl::KeyBuffer<T>& buf, int dst_worker_id,
                   int tag, MPI_Comm comm) {
    Send(buf.buffer(), dst_worker_id, tag, comm);
  }

  static void recv(id_indexer_impl::KeyBuffer<T>& buf, int src_worker_id,
                   int tag, MPI_Comm comm) {
    Recv(buf.buffer(), src_worker_id, tag, comm);
  }
};

template <>
struct CommImpl<id_indexer_impl::KeyBuffer<nonstd::string_view>> {
  static void send(const id_indexer_impl::KeyBuffer<nonstd::string_view>& buf,
                   int dst_worker_id, int tag, MPI_Comm comm) {
    Send(buf.buffer(), dst_worker_id, tag, comm);
  }

  static void recv(id_indexer_impl::KeyBuffer<nonstd::string_view>& buf,
                   int src_worker_id, int tag, MPI_Comm comm) {
    Recv(buf.buffer(), src_worker_id, tag, comm);
  }
};

}  // namespace sync_comm

template <typename KEY_T, typename INDEX_T>
class IdIndexer {
 public:
  using key_buffer_t = typename id_indexer_impl::KeyBuffer<KEY_T>;
  using ind_buffer_t = std::vector<INDEX_T, Allocator<INDEX_T>>;
  using dist_buffer_t = std::vector<int8_t, Allocator<int8_t>>;

  IdIndexer() : hasher_() { reset_to_empty_state(); }
  IdIndexer(IdIndexer&& rhs) { swap(rhs); }
  ~IdIndexer() = default;

  IdIndexer& operator=(IdIndexer&& rhs) {
    swap(rhs);
    return *this;
  }

  size_t entry_num() const { return distances_.size(); }

  size_t memory_usage() const {
    size_t ret = keys_.dump_size();
    ret += indices_.size() * sizeof(INDEX_T);
    ret += distances_.size() * sizeof(int8_t);
    return ret;
  }

  bool add(const KEY_T& oid, INDEX_T& lid) {
    size_t index =
        hash_policy_.index_for_hash(hasher_(oid), num_slots_minus_one_);

    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      INDEX_T cur_lid = indices_[index];
      if (keys_.get(cur_lid) == oid) {
        lid = cur_lid;
        return false;
      }
    }

    lid = static_cast<INDEX_T>(keys_.size());
    keys_.push_back(oid);
    assert(keys_.size() == num_elements_ + 1);
    emplace_new_value(distance_from_desired, index, lid);
    assert(keys_.size() == num_elements_);
    return true;
  }

  bool add(KEY_T&& oid, INDEX_T& lid) {
    size_t index =
        hash_policy_.index_for_hash(hasher_(oid), num_slots_minus_one_);

    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      INDEX_T cur_lid = indices_[index];
      if (keys_.get(cur_lid) == oid) {
        lid = cur_lid;
        return false;
      }
    }

    lid = static_cast<INDEX_T>(keys_.size());
    keys_.push_back(std::move(oid));
    assert(keys_.size() == num_elements_ + 1);
    emplace_new_value(distance_from_desired, index, lid);
    assert(keys_.size() == num_elements_);
    return true;
  }

  bool _add(const KEY_T& oid, size_t hash_value, INDEX_T& lid) {
    size_t index =
        hash_policy_.index_for_hash(hash_value, num_slots_minus_one_);

    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      INDEX_T cur_lid = indices_[index];
      if (keys_.get(cur_lid) == oid) {
        lid = cur_lid;
        return false;
      }
    }

    lid = static_cast<INDEX_T>(keys_.size());
    keys_.push_back(oid);
    assert(keys_.size() == num_elements_ + 1);
    emplace_new_value(distance_from_desired, index, lid);
    assert(keys_.size() == num_elements_);
    return true;
  }

  bool _add(KEY_T&& oid, size_t hash_value, INDEX_T& lid) {
    size_t index =
        hash_policy_.index_for_hash(hash_value, num_slots_minus_one_);

    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      INDEX_T cur_lid = indices_[index];
      if (keys_.get(cur_lid) == oid) {
        lid = cur_lid;
        return false;
      }
    }

    lid = static_cast<INDEX_T>(keys_.size());
    keys_.push_back(std::move(oid));
    assert(keys_.size() == num_elements_ + 1);
    emplace_new_value(distance_from_desired, index, lid);
    assert(keys_.size() == num_elements_);
    return true;
  }

  void _add(const KEY_T& oid) {
    size_t index =
        hash_policy_.index_for_hash(hasher_(oid), num_slots_minus_one_);

    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      if (keys_.get(indices_[index]) == oid) {
        return;
      }
    }

    INDEX_T lid = static_cast<INDEX_T>(keys_.size());
    keys_.push_back(oid);
    assert(keys_.size() == num_elements_ + 1);
    emplace_new_value(distance_from_desired, index, lid);
    assert(keys_.size() == num_elements_);
  }

  void _add(KEY_T&& oid) {
    size_t index =
        hash_policy_.index_for_hash(hasher_(oid), num_slots_minus_one_);

    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      if (keys_.get(indices_[index]) == oid) {
        return;
      }
    }

    INDEX_T lid = static_cast<INDEX_T>(keys_.size());
    keys_.push_back(std::move(oid));
    assert(keys_.size() == num_elements_ + 1);
    emplace_new_value(distance_from_desired, index, lid);
    assert(keys_.size() == num_elements_);
  }

  size_t bucket_count() const {
    return num_slots_minus_one_ ? num_slots_minus_one_ + 1 : 0;
  }

  bool empty() const { return (num_elements_ == 0); }

  size_t size() const { return num_elements_; }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    if (lid >= num_elements_) {
      return false;
    }
    oid = keys_.get(lid);
    return true;
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    size_t index =
        hash_policy_.index_for_hash(hasher_(oid), num_slots_minus_one_);
    for (int8_t distance = 0; distances_[index] >= distance;
         ++distance, ++index) {
      INDEX_T ret = indices_[index];
      if (keys_.get(ret) == oid) {
        lid = ret;
        return true;
      }
    }
    return false;
  }

  bool _get_index(const KEY_T& oid, size_t hash, INDEX_T& lid) const {
    size_t index = hash_policy_.index_for_hash(hash, num_slots_minus_one_);
    for (int8_t distance = 0; distances_[index] >= distance;
         ++distance, ++index) {
      INDEX_T ret = indices_[index];
      if (keys_.get(ret) == oid) {
        lid = ret;
        return true;
      }
    }
    return false;
  }

  void swap(IdIndexer<KEY_T, INDEX_T>& rhs) {
    keys_.swap(rhs.keys_);
    indices_.swap(rhs.indices_);
    distances_.swap(rhs.distances_);

    hash_policy_.swap(rhs.hash_policy_);
    std::swap(max_lookups_, rhs.max_lookups_);
    std::swap(num_elements_, rhs.num_elements_);
    std::swap(num_slots_minus_one_, rhs.num_slots_minus_one_);

    std::swap(hasher_, rhs.hasher_);
  }

  const key_buffer_t& keys() const { return keys_; }

  key_buffer_t& keys() { return keys_; }

  template <typename IOADAPTOR_T>
  void Serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    keys_.serialize(writer);

    size_t mod_function_index = hash_policy_.get_mod_function_index();
    int8_t max_lookups_val = max_lookups_;
    size_t num_elements_val = num_elements_;
    size_t num_slots_minus_one_val = num_slots_minus_one_;
    CHECK(writer->Write(&mod_function_index, sizeof(size_t)));
    CHECK(writer->Write(&max_lookups_val, sizeof(int8_t)));
    CHECK(writer->Write(&num_elements_val, sizeof(size_t)));
    CHECK(writer->Write(&num_slots_minus_one_val, sizeof(size_t)));

    size_t indices_size = indices_.size();
    CHECK(writer->Write(&indices_size, sizeof(size_t)));
    if (indices_.size() > 0) {
      CHECK(writer->Write(const_cast<INDEX_T*>(indices_.data()),
                          indices_size * sizeof(INDEX_T)));
    }
    size_t distances_size = distances_.size();
    CHECK(writer->Write(&distances_size, sizeof(size_t)));
    if (distances_.size() > 0) {
      CHECK(writer->Write(const_cast<int8_t*>(distances_.data()),
                          distances_size * sizeof(int8_t)));
    }
  }

  template <typename IOADAPTOR_T>
  void Deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    keys_.deserialize(reader);

    size_t mod_function_index;
    CHECK(reader->Read(&mod_function_index, sizeof(size_t)));
    hash_policy_.set_mod_function_by_index(mod_function_index);
    CHECK(reader->Read(&max_lookups_, sizeof(int8_t)));
    CHECK(reader->Read(&num_elements_, sizeof(size_t)));
    CHECK(reader->Read(&num_slots_minus_one_, sizeof(size_t)));

    size_t indices_size;
    CHECK(reader->Read(&indices_size, sizeof(size_t)));
    indices_.resize(indices_size);
    if (indices_size > 0) {
      CHECK(reader->Read(indices_.data(), indices_.size() * sizeof(INDEX_T)));
    }

    size_t distances_size;
    CHECK(reader->Read(&distances_size, sizeof(size_t)));
    distances_.resize(distances_size);
    if (distances_size > 0) {
      CHECK(
          reader->Read(distances_.data(), distances_.size() * sizeof(int8_t)));
    }
  }

  void serialize_to_mem(std::vector<char>& buf) const {
    keys_.serialize_to_mem(buf);
    size_t mod_function_index = hash_policy_.get_mod_function_index();
    encode_val(mod_function_index, buf);
    encode_val(max_lookups_, buf);
    encode_val(num_elements_, buf);
    encode_val(num_slots_minus_one_, buf);

    encode_vec(indices_, buf);
    encode_vec(distances_, buf);
  }

 private:
  void emplace(INDEX_T lid) {
    KEY_T key = keys_.get(lid);
    size_t index =
        hash_policy_.index_for_hash(hasher_(key), num_slots_minus_one_);
    int8_t distance_from_desired = 0;
    for (; distances_[index] >= distance_from_desired;
         ++index, ++distance_from_desired) {
      if (indices_[index] == lid) {
        return;
      }
    }

    emplace_new_value(distance_from_desired, index, lid);
  }

  void emplace_new_value(int8_t distance_from_desired, size_t index,
                         INDEX_T lid) {
    if (num_slots_minus_one_ == 0 || distance_from_desired == max_lookups_ ||
        num_elements_ + 1 >
            (num_slots_minus_one_ + 1) * id_indexer_impl::max_load_factor) {
      grow();
      return;
    } else if (distances_[index] < 0) {
      indices_[index] = lid;
      distances_[index] = distance_from_desired;
      ++num_elements_;
      return;
    }
    INDEX_T to_insert = lid;
    std::swap(distance_from_desired, distances_[index]);
    std::swap(to_insert, indices_[index]);
    for (++distance_from_desired, ++index;; ++index) {
      if (distances_[index] < 0) {
        indices_[index] = to_insert;
        distances_[index] = distance_from_desired;
        ++num_elements_;
        return;
      } else if (distances_[index] < distance_from_desired) {
        std::swap(distance_from_desired, distances_[index]);
        std::swap(to_insert, indices_[index]);
        ++distance_from_desired;
      } else {
        ++distance_from_desired;
        if (distance_from_desired == max_lookups_) {
          grow();
          return;
        }
      }
    }
  }

  void grow() { rehash(std::max(size_t(4), 2 * bucket_count())); }

  void rehash(size_t num_buckets) {
    num_buckets = std::max(
        num_buckets, static_cast<size_t>(std::ceil(
                         num_elements_ / id_indexer_impl::max_load_factor)));

    if (num_buckets == 0) {
      reset_to_empty_state();
      return;
    }

    auto new_prime_index = hash_policy_.next_size_over(num_buckets);
    if (num_buckets == bucket_count()) {
      return;
    }

    int8_t new_max_lookups = compute_max_lookups(num_buckets);

    dist_buffer_t new_distances(num_buckets + new_max_lookups);
    ind_buffer_t new_indices(num_buckets + new_max_lookups);

    size_t special_end_index = num_buckets + new_max_lookups - 1;
    for (size_t i = 0; i != special_end_index; ++i) {
      new_distances[i] = -1;
    }
    new_distances[special_end_index] = 0;

    new_indices.swap(indices_);
    new_distances.swap(distances_);

    std::swap(num_slots_minus_one_, num_buckets);
    --num_slots_minus_one_;
    hash_policy_.commit(new_prime_index);

    max_lookups_ = new_max_lookups;

    num_elements_ = 0;
    INDEX_T elem_num = static_cast<INDEX_T>(keys_.size());
    for (INDEX_T lid = 0; lid < elem_num; ++lid) {
      emplace(lid);
    }
  }

  void reset_to_empty_state() {
    keys_.clear();

    indices_.clear();
    distances_.clear();
    indices_.resize(id_indexer_impl::min_lookups);
    distances_.resize(id_indexer_impl::min_lookups, -1);
    distances_[id_indexer_impl::min_lookups - 1] = 0;

    num_slots_minus_one_ = 0;
    hash_policy_.reset();
    max_lookups_ = id_indexer_impl::min_lookups - 1;
    num_elements_ = 0;
  }

  static int8_t compute_max_lookups(size_t num_buckets) {
    int8_t desired = ska::detailv3::log2(num_buckets);
    return std::max(id_indexer_impl::min_lookups, desired);
  }

  template <typename _T, typename _Enable>
  friend struct sync_comm::CommImpl;

  key_buffer_t keys_;
  ind_buffer_t indices_;
  dist_buffer_t distances_;

  ska::ska::prime_number_hash_policy hash_policy_;
  int8_t max_lookups_ = id_indexer_impl::min_lookups - 1;
  size_t num_elements_ = 0;
  size_t num_slots_minus_one_ = 0;

  std::hash<KEY_T> hasher_;
};

template <typename KEY_T, typename INDEX_T>
class IdIndexerView {
 public:
  IdIndexerView() : hasher_() {}
  ~IdIndexerView() = default;

  void Init(const void* data, size_t size) {
    const char* ptr = reinterpret_cast<const char*>(data);
    size_t cur = keys_.init(ptr, size);
    ptr += cur;

    size_t mod_function_index;
    ptr = decode_val(mod_function_index, ptr);
    hash_policy_.set_mod_function_by_index(mod_function_index);

    ptr = decode_val(max_lookups_, ptr);
    ptr = decode_val(num_elements_, ptr);
    ptr = decode_val(num_slots_minus_one_, ptr);

    size_t used_size = ptr - reinterpret_cast<const char*>(data);
    size -= used_size;

    cur = indices_.init(ptr, size);
    ptr += cur;
    size -= cur;

    distances_.init(ptr, size);
  }

  size_t entry_num() const { return distances_.size(); }

  size_t bucket_count() const {
    return num_slots_minus_one_ ? num_slots_minus_one_ + 1 : 0;
  }

  size_t size() const { return num_elements_; }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    if (lid >= num_elements_) {
      return false;
    }
    oid = keys_.get(lid);
    return true;
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    size_t index =
        hash_policy_.index_for_hash(hasher_(oid), num_slots_minus_one_);
    for (int8_t distance = 0; distances_.get(index) >= distance;
         ++distance, ++index) {
      INDEX_T ret = indices_.get(index);
      if (keys_.get(ret) == oid) {
        lid = ret;
        return true;
      }
    }
    return false;
  }

  bool _get_index(const KEY_T& oid, size_t hash, INDEX_T& lid) const {
    size_t index = hash_policy_.index_for_hash(hash, num_slots_minus_one_);
    for (int8_t distance = 0; distances_.get(index) >= distance;
         ++distance, ++index) {
      INDEX_T ret = indices_.get(index);
      if (keys_.get(ret) == oid) {
        lid = ret;
        return true;
      }
    }
    return false;
  }

 private:
  typename id_indexer_impl::KeyBufferView<KEY_T> keys_;
  ref_vector<INDEX_T> indices_;
  ref_vector<int8_t> distances_;

  ska::ska::prime_number_hash_policy hash_policy_;
  int8_t max_lookups_ = id_indexer_impl::min_lookups - 1;
  size_t num_elements_ = 0;
  size_t num_slots_minus_one_ = 0;

  std::hash<KEY_T> hasher_;
};

namespace sync_comm {

template <typename OID_T, typename VID_T>
struct CommImpl<IdIndexer<OID_T, VID_T>> {
  static void send(const IdIndexer<OID_T, VID_T>& indexer, int dst_worker_id,
                   int tag, MPI_Comm comm) {
    InArchive arc;
    arc << indexer.hash_policy_.get_mod_function_index() << indexer.max_lookups_
        << indexer.num_elements_ << indexer.num_slots_minus_one_;
    Send(arc, dst_worker_id, tag, comm);
    Send(indexer.keys_, dst_worker_id, tag, comm);
    Send(indexer.indices_, dst_worker_id, tag, comm);
    Send(indexer.distances_, dst_worker_id, tag, comm);
  }

  static void recv(IdIndexer<OID_T, VID_T>& indexer, int src_worker_id, int tag,
                   MPI_Comm comm) {
    OutArchive arc;
    Recv(arc, src_worker_id, tag, comm);
    size_t mod_function_index;
    arc >> mod_function_index >> indexer.max_lookups_ >>
        indexer.num_elements_ >> indexer.num_slots_minus_one_;
    indexer.hash_policy_.set_mod_function_by_index(mod_function_index);
    Recv(indexer.keys_, src_worker_id, tag, comm);
    Recv(indexer.indices_, src_worker_id, tag, comm);
    Recv(indexer.distances_, src_worker_id, tag, comm);
  }
};

}  // namespace sync_comm

}  // namespace grape

#endif  // GRAPE_GRAPH_ID_INDEXER_H_
