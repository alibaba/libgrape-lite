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

#ifndef GRAPE_GRAPH_PERFECT_HASH_INDEXER_H_
#define GRAPE_GRAPH_PERFECT_HASH_INDEXER_H_

#include "grape/graph/hashmap_indexer_impl.h"
#include "grape/util.h"
#include "grape/utils/pthash_utils/single_phf_view.h"
#include "grape/utils/string_view_vector.h"

namespace grape {

template <typename KEY_T, typename INDEX_T>
class PHIdxerView {
 public:
  PHIdxerView() {}
  ~PHIdxerView() {}

  void init(const void* buffer, size_t size) {
    mem_loader loader(reinterpret_cast<const char*>(buffer), size);
    phf_view_.load(loader);
    keys_view_.load(loader);
  }

  size_t entry_num() const { return keys_view_.size(); }

  bool empty() const { return keys_view_.empty(); }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    if (lid >= keys_view_.size()) {
      return false;
    }
    oid = keys_view_.get(lid);
    return true;
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    auto idx = phf_view_(oid);
    if (idx < keys_view_.size() && keys_view_.get(idx) == oid) {
      lid = idx;
      return true;
    }
    return false;
  }

  size_t size() const { return keys_view_.size(); }

 private:
  SinglePHFView<murmurhasher> phf_view_;
  hashmap_indexer_impl::KeyBufferView<KEY_T> keys_view_;
};

template <typename KEY_T, typename INDEX_T>
class ImmPHIdxer {
 public:
  void Init(std::vector<char>&& buf) {
    buffer_ = std::move(buf);
    idxer_.init(buffer_.data(), buffer_.size());
  }

  void Init(const char *buf, size_t size) {
    idxer_.init(buf, size);
  }

  size_t entry_num() const { return idxer_.entry_num(); }

  bool empty() const { return idxer_.empty(); }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    return idxer_.get_key(lid, oid);
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    return idxer_.get_index(oid, lid);
  }

  size_t size() const { return idxer_.size(); }

  const std::vector<char>& buffer() const { return buffer_; }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& path) const {
    auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(path));
    io_adaptor->Open("wb");
    CHECK(io_adaptor->Write(const_cast<char*>(buffer_.data()), buffer_.size()));
    io_adaptor->Close();
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& path) {
    size_t fsize = file_size(path);
    auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(path));
    buffer_.resize(fsize);
    io_adaptor->Open();
    CHECK(io_adaptor->Read(buffer_.data(), buffer_.size()));
    io_adaptor->Close();
    idxer_.init(buffer_.data(), buffer_.size());
  }

 private:
  std::vector<char> buffer_;
  PHIdxerView<KEY_T, INDEX_T> idxer_;
};

template <typename KEY_T, typename INDEX_T>
class PHIdxerViewBuilder {
 public:
  PHIdxerViewBuilder() = default;
  ~PHIdxerViewBuilder() = default;

  void add(const KEY_T& oid) { keys_.push_back(oid); }

  void add(KEY_T&& oid) { keys_.push_back(std::move(oid)); }

  void buildPhf() {
    SinglePHFView<murmurhasher>::build(keys_.begin(),
                                                     keys_.size(), phf, 1);
    std::vector<KEY_T> ordered_keys(keys_.size());
    for (auto& key : keys_) {
      size_t idx = phf(key);
      ordered_keys[idx] = key;
    }
    for (auto& key : ordered_keys) {
      key_buffer.push_back(key);
    }
  }

  void finish(void *buffer, size_t size, ImmPHIdxer<KEY_T, INDEX_T> &idxer) {
    external_mem_dumper dumper(buffer, size);
    phf.dump(dumper);
    key_buffer.dump(dumper);
    idxer.Init(static_cast<const char*>(dumper.buffer()), dumper.size());
  }

  size_t getSerializeSize() {
    return phf.num_bits() / 8 + key_buffer.dump_size();
  }

 private:
  std::vector<KEY_T> keys_;
  hashmap_indexer_impl::KeyBuffer<KEY_T> key_buffer;
  pthash::single_phf<murmurhasher, pthash::dictionary_dictionary, true> phf;
};

}  // namespace grape

#endif  // GRAPE_GRAPH_PERFECT_HASH_INDEXER_H_