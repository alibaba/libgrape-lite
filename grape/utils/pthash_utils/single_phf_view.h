/** Copyright 2020 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRAPE_UTILS_PTHASH_UTILS_SINGLE_PHF_VIEW_H_
#define GRAPE_UTILS_PTHASH_UTILS_SINGLE_PHF_VIEW_H_

#include "grape/utils/pthash_utils/encoders_view.h"
#include "pthash/builders/util.hpp"
#include "pthash/encoders/encoders.hpp"
#include "pthash/single_phf.hpp"
#include "pthash/utils/bucketers.hpp"
#include "pthash/utils/hasher.hpp"

namespace grape {

struct mem_dumper {
 public:
  mem_dumper() = default;
  ~mem_dumper() = default;

  template <typename T>
  void dump(const T& val) {
    static_assert(std::is_pod<T>::value, "T must be POD type");
    const char* ptr = reinterpret_cast<const char*>(&val);
    buf_.insert(buf_.end(), ptr, ptr + sizeof(T));
  }

  template <typename T, typename ALLOC_T>
  void dump_vec(const std::vector<T, ALLOC_T>& vec) {
    static_assert(std::is_pod<T>::value, "T must be POD type");
    size_t n = vec.size();
    dump(n);
    const char* ptr = reinterpret_cast<const char*>(vec.data());
    buf_.insert(buf_.end(), ptr, ptr + sizeof(T) * n);
  }

  const std::vector<char>& buffer() const { return buf_; }
  std::vector<char>& buffer() { return buf_; }

  size_t size() const { return buf_.size(); }

 private:
  std::vector<char> buf_;
};

struct external_mem_dumper {
 public:
  external_mem_dumper(void* buf, size_t size) : buf_(buf), size_(size) {}

  ~external_mem_dumper() = default;

  template <typename T>
  void dump(const T& val) {
    static_assert(std::is_pod<T>::value, "T must be POD type");
    const char* ptr = reinterpret_cast<const char*>(&val);
    if (pos_ + sizeof(T) > size_) {
      return;
    }
    memcpy(reinterpret_cast<char*>(buf_) + pos_, ptr, sizeof(T));
    pos_ += sizeof(T);
  }

  template <typename T, typename ALLOC_T>
  void dump_vec(const std::vector<T, ALLOC_T>& vec) {
    static_assert(std::is_pod<T>::value, "T must be POD type");
    size_t n = vec.size();
    if (pos_ + sizeof(T) * n + sizeof(size_t) > size_) {
      return;
    }
    dump(n);
    const char* ptr = reinterpret_cast<const char*>(vec.data());
    memcpy(reinterpret_cast<char*>(buf_) + pos_, ptr, sizeof(T) * n);
    pos_ += sizeof(T) * n;
  }

  const void* buffer() const { return buf_; }

  size_t size() const { return size_; }

 private:
  void* buf_ = nullptr;
  size_t pos_ = 0;
  size_t size_ = 0;
};

struct mem_loader {
 public:
  mem_loader(const char* buf, size_t size)
      : begin_(buf), ptr_(buf), end_(buf + size) {}
  ~mem_loader() = default;

  template <typename T>
  void load(T& val) {
    memcpy(&val, ptr_, sizeof(T));
    ptr_ += sizeof(T);
  }

  template <typename T>
  void load_vec(std::vector<T>& vec) {
    static_assert(std::is_pod<T>::value, "T must be POD type");
    size_t n;
    load(n);
    vec.resize(n);
    memcpy(vec.data(), ptr_, n * sizeof(T));
    ptr_ += (n * sizeof(T));
  }

  template <typename T>
  void load_ref_vec(ref_vector<T>& vec) {
    ptr_ += vec.init(ptr_, end_ - ptr_);
  }

  const char* data() const { return ptr_; }
  size_t remaining() const { return end_ - ptr_; }
  size_t used() const { return ptr_ - begin_; }

 private:
  const char* begin_;
  const char* ptr_;
  const char* end_;
};

// This code is an adaptation from
// https://github.com/jermp/pthash/blob/master/include/single_phf.hpp
template <typename Hasher>
struct SinglePHFView {
 public:
  SinglePHFView() = default;
  ~SinglePHFView() = default;

  template <typename T>
  uint64_t operator()(T const& key) const {
    auto hash = Hasher::hash(key, m_seed);
    return position(hash);
  }

  uint64_t position(typename Hasher::hash_type hash) const {
    uint64_t bucket = m_bucketer.bucket(hash.first());
    uint64_t pilot = m_pilots.access(bucket);
    uint64_t hashed_pilot = pthash::default_hash64(pilot, m_seed);
    uint64_t p =
        fastmod::fastmod_u64(hash.second() ^ hashed_pilot, m_M, m_table_size);
    if (PTHASH_LIKELY(p < m_num_keys))
      return p;
    return m_free_slots.access(p - m_num_keys);
  }

  template <typename Loader>
  void load(Loader& loader) {
    loader.load(m_seed);
    loader.load(m_num_keys);
    loader.load(m_table_size);
    loader.load(m_M);
    m_bucketer.load(loader);
    m_pilots.load(loader);
    m_free_slots.load(loader);
  }

  template <typename Iterator, typename Dumper>
  static void build(Iterator keys, uint64_t n, Dumper& dumper, int thread_num) {
    pthash::build_configuration config;
    config.c = 7.0;
    config.alpha = 0.94;
    config.num_threads = thread_num;
    config.minimal_output = true;
    config.verbose_output = false;

    pthash::single_phf<murmurhasher, pthash::dictionary_dictionary, true> phf;
    phf.build_in_internal_memory(keys, n, config);
    std::set<size_t> idx;
    for (uint64_t k = 0; k < n; ++k) {
      idx.insert(phf(*keys));
      ++keys;
    }
    phf.dump(dumper);
  }

  template <typename Iterator>
  static void build(
      Iterator keys, uint64_t n,
      pthash::single_phf<murmurhasher, pthash::dictionary_dictionary, true>&
          phf,
      int thread_num) {
    pthash::build_configuration config;
    config.c = 7.0;
    config.alpha = 0.94;
    config.num_threads = thread_num;
    config.minimal_output = true;
    config.verbose_output = false;

    phf.build_in_internal_memory(keys, n, config);
  }

 private:
  uint64_t m_seed;
  uint64_t m_num_keys;
  uint64_t m_table_size;
  __uint128_t m_M;
  pthash::skew_bucketer m_bucketer;
  dual_dictionary_view m_pilots;
  ef_sequence_view m_free_slots;
};

}  // namespace grape

#endif  // GRAPE_UTILS_PTHASH_UTILS_SINGLE_PHF_VIEW_H_
