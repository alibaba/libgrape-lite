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

#ifndef GRAPE_VERTEX_MAP_IDXERS_PTHASH_IDXER_H_
#define GRAPE_VERTEX_MAP_IDXERS_PTHASH_IDXER_H_

#include "grape/utils/gcontainer.h"
#include "grape/utils/pthash_utils/ph_indexer_view.h"

namespace grape {

template <typename OID_T, typename VID_T>
class PTHashIdxer : public IdxerBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  PTHashIdxer() {}
  explicit PTHashIdxer(Array<char, Allocator<char>>&& buf)
      : buffer_(std::move(buf)) {
    idxer_.init(buffer_.data(), buffer_.size());
  }
  ~PTHashIdxer() {}

  void Init(void* buffer, size_t size) { idxer_.init(buffer, size); }

  bool get_key(VID_T vid, internal_oid_t& oid) const override {
    return idxer_.get_key(vid, oid);
  }

  bool get_index(const internal_oid_t& oid, VID_T& vid) const override {
    return idxer_.get_index(oid, vid);
  }

  IdxerType type() const override { return IdxerType::kPTHashIdxer; }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    idxer_.Serialize(writer);
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    size_t size;
    CHECK(reader->Read(&size, sizeof(size_t)));
    if (size > 0) {
      buffer_.resize(size);
      CHECK(reader->Read(buffer_.data(), size));
      idxer_.init(buffer_.data(), size);
    }
  }

  size_t size() const override { return idxer_.size(); }

  size_t memory_usage() const override { return buffer_.size(); }

 private:
  Array<char, Allocator<char>> buffer_;
  PHIndexerView<internal_oid_t, VID_T> idxer_;
};

template <typename OID_T, typename VID_T>
class PTHashIdxerDummyBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override {}

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    sync_comm::Recv(buffer_, target, tag, comm_spec.comm());
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    LOG(ERROR) << "PTHashIdxerDummyBuilder should not be used to sync response";
  }

  IdxerBase<OID_T, VID_T>* finish() override {
    return new PTHashIdxer<OID_T, VID_T>(std::move(buffer_));
  }

 private:
  Array<char, Allocator<char>> buffer_;
};

template <typename OID_T, typename VID_T>
class PTHashIdxerBuilder : public IdxerBuilderBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  PTHashIdxerBuilder() {}
  ~PTHashIdxerBuilder() {}

  void add(const internal_oid_t& oid) override { keys_.push_back(OID_T(oid)); }

  void buildPhf() {
    if (build_phf_) {
      return;
    }
    DistinctSort(keys_);
    SinglePHFView<murmurhasher>::build(keys_.begin(), keys_.size(), phf_, 1);
    std::vector<OID_T> ordered_keys(keys_.size());
    for (auto& key : keys_) {
      size_t idx = phf_(key);
      ordered_keys[idx] = key;
    }
    key_buffer_.clear();
    for (auto& key : ordered_keys) {
      key_buffer_.push_back(key);
    }
    build_phf_ = true;
  }

  size_t getSerializeSize() {
    return phf_.num_bits() / 8 + key_buffer_.dump_size();
  }

  /*
   * Finish building the perfect hash index in a allocated buffer.
   * After add all keys, call buildPhf to build the perfect hash function.
   * And then allocate a buffer with getSerializeSize() bytes.
   * Call finishInplace to finish building the index in the buffer.
   */
  void finishInplace(void* buffer, size_t size,
                     PTHashIdxer<OID_T, VID_T>& idxer) {
    external_mem_dumper dumper(reinterpret_cast<char*>(buffer), size);
    phf_.dump(dumper);
    key_buffer_.dump(dumper);
    idxer.Init(buffer, size);
  }

  /*
   * Finish building the perfect hash index in an internal
   * buffer(std::vector<char>). After add all keys, call finish to build the
   * perfect hash index and serialize it.
   */
  IdxerBase<OID_T, VID_T>* finish() override {
    buildPhf();
    if (getSerializeSize() != buffer_.size()) {
      buffer_.resize(getSerializeSize());
      external_mem_dumper dumper(buffer_.data(), buffer_.size());
      phf_.dump(dumper);
      key_buffer_.dump(dumper);
    }
    auto idxer = new PTHashIdxer<OID_T, VID_T>(std::move(buffer_));
    return idxer;
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    LOG(ERROR) << "PTHashIdxerBuilder should not be used to sync request";
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    buildPhf();
    if (getSerializeSize() != buffer_.size()) {
      buffer_.resize(getSerializeSize());
      external_mem_dumper dumper(buffer_.data(), buffer_.size());
      phf_.dump(dumper);
      key_buffer_.dump(dumper);
    }

    sync_comm::Send(buffer_, source, tag, comm_spec.comm());
  }

 private:
  std::vector<OID_T> keys_;
  id_indexer_impl::KeyBuffer<internal_oid_t> key_buffer_;
  pthash::single_phf<murmurhasher, pthash::dictionary_dictionary, true> phf_;

  Array<char, Allocator<char>> buffer_;
  bool build_phf_ = false;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_PTHASH_IDXER_H_
