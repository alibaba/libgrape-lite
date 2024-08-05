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

#ifndef GRAPE_VERTEX_MAP_IDXERS_HASHMAP_IDXER_H_
#define GRAPE_VERTEX_MAP_IDXERS_HASHMAP_IDXER_H_

#include "grape/vertex_map/idxers/idxer_base.h"

namespace grape {

template <typename OID_T, typename VID_T>
class HashMapIdxer : public IdxerBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  HashMapIdxer() {}
  explicit HashMapIdxer(IdIndexer<internal_oid_t, VID_T>&& indexer)
      : indexer_(std::move(indexer)) {}

  bool get_key(VID_T vid, internal_oid_t& oid) const override {
    return indexer_.get_key(vid, oid);
  }

  bool get_index(const internal_oid_t& oid, VID_T& vid) const override {
    return indexer_.get_index(oid, vid);
  }

  IdxerType type() const override { return IdxerType::kHashMapIdxer; }

  void serialize(IOAdaptorBase* writer) override { indexer_.Serialize(writer); }
  void deserialize(IOAdaptorBase* reader) override {
    indexer_.Deserialize(reader);
  }

  size_t size() const override { return indexer_.size(); }

  void add(const internal_oid_t& oid) { indexer_._add(oid); }

 private:
  IdIndexer<internal_oid_t, VID_T> indexer_;
};

template <typename OID_T, typename VID_T>
class HashMapIdxerBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override { indexer_._add(oid); }

  IdxerBase<OID_T, VID_T>* finish() override {
    return new HashMapIdxer<OID_T, VID_T>(std::move(indexer_));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    LOG(ERROR) << "HashMapIdxerBuilder should not be used to sync request";
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    int req_type;
    sync_comm::Recv(req_type, source, tag, comm_spec.comm());
    if (req_type == 0) {
      // request all
      sync_comm::Send(indexer_, source, tag + 1, comm_spec.comm());
    } else if (req_type == 1) {
      // request partial
      typename IdIndexer<OID_T, VID_T>::key_buffer_t keys;
      sync_comm::Recv(keys, source, tag, comm_spec.comm());
      std::vector<VID_T> response;
      for (auto& key : keys) {
        VID_T vid;
        if (indexer_.get_index(key, vid)) {
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
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_HASHMAP_IDXER_H_