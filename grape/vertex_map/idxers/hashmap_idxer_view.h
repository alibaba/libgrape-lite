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

#ifndef GRAPE_VERTEX_MAP_IDXERS_HASHMAP_IDXER_VIEW_H_
#define GRAPE_VERTEX_MAP_IDXERS_HASHMAP_IDXER_VIEW_H_

#include "grape/graph/id_indexer.h"
#include "grape/vertex_map/idxers/idxer_base.h"

namespace grape {

template <typename OID_T, typename VID_T>
class HashMapIdxerView : public IdxerBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  HashMapIdxerView() {}
  explicit HashMapIdxerView(Array<char, Allocator<char>>&& buf)
      : buffer_(std::move(buf)) {
    indexer_.Init(buffer_.data(), buffer_.size());
  }
  ~HashMapIdxerView() {}

  bool get_key(VID_T vid, internal_oid_t& oid) const override {
    return indexer_.get_key(vid, oid);
  }

  bool get_index(const internal_oid_t& oid, VID_T& vid) const override {
    return indexer_.get_index(oid, vid);
  }

  IdxerType type() const override { return IdxerType::kHashMapIdxerView; }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    size_t size = buffer_.size();
    writer->Write(&size, sizeof(size_t));
    if (size > 0) {
      writer->Write(buffer_.data(), size);
    }
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    size_t size;
    CHECK(reader->Read(&size, sizeof(size_t)));
    if (size > 0) {
      buffer_.resize(size);
      CHECK(reader->Read(buffer_.data(), size));
      indexer_.Init(buffer_.data(), size);
    }
  }

  size_t size() const override { return indexer_.size(); }

  size_t memory_usage() const override { return buffer_.size(); }

 private:
  IdIndexerView<internal_oid_t, VID_T> indexer_;
  Array<char, Allocator<char>> buffer_;
};

template <typename OID_T, typename VID_T>
class HashMapIdxerViewDummyBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override {}

  std::unique_ptr<IdxerBase<OID_T, VID_T>> finish() override {
    return std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new HashMapIdxerView<OID_T, VID_T>(std::move(buffer_)));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    int req_type = 0;
    sync_comm::Send(req_type, target, tag, comm_spec.comm());
    sync_comm::Recv(buffer_, target, tag + 1, comm_spec.comm());
  }
  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    LOG(ERROR)
        << "HashMapIdxerViewDummyBuilder should not be used to sync response";
  }

 private:
  Array<char, Allocator<char>> buffer_;
};

template <typename OID_T, typename VID_T>
class HashMapIdxerViewBuilder : public IdxerBuilderBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  HashMapIdxerViewBuilder() {}
  ~HashMapIdxerViewBuilder() {}

  void add(const internal_oid_t& oid) override { indexer_._add(oid); }

  std::unique_ptr<IdxerBase<OID_T, VID_T>> finish() override {
    if (buffer_.empty() && indexer_.size() > 0) {
      indexer_.serialize_to_mem(buffer_);
    }
    Array<char, Allocator<char>> buffer;
    buffer.resize(buffer_.size());
    memcpy(buffer.data(), buffer_.data(), buffer_.size());
    return std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new HashMapIdxerView<OID_T, VID_T>(std::move(buffer)));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    LOG(ERROR) << "HashMapIdxerBuilder should not be used to sync request";
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    int req_type;
    sync_comm::Recv(req_type, source, tag, comm_spec.comm());
    if (req_type == 0) {
      // request all
      if (buffer_.empty() && indexer_.size() > 0) {
        indexer_.serialize_to_mem(buffer_);
      }
      sync_comm::Send(buffer_, source, tag + 1, comm_spec.comm());
    } else if (req_type == 1) {
      // request partial
      typename IdIndexer<OID_T, VID_T>::key_buffer_t keys;
      sync_comm::Recv(keys, source, tag, comm_spec.comm());
      std::vector<VID_T> response;
      size_t keys_num = keys.size();
      for (size_t i = 0; i < keys_num; ++i) {
        VID_T vid;
        if (indexer_.get_index(keys.get(i), vid)) {
          response.push_back(vid);
        } else {
          response.push_back(std::numeric_limits<VID_T>::max());
        }
      }
      sync_comm::Send(response, source, tag + 1, comm_spec.comm());
    }
  }

 private:
  IdIndexer<internal_oid_t, VID_T> indexer_;
  std::vector<char> buffer_;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_HASHMAP_IDXER_VIEW_H_
