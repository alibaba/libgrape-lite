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

#ifndef GRAPE_VERTEX_MAP_IDXERS_LOCAL_IDXER_H_
#define GRAPE_VERTEX_MAP_IDXERS_LOCAL_IDXER_H_

#include "grape/vertex_map/idxers/idxer_base.h"

namespace grape {

template <typename OID_T, typename VID_T>
class LocalIdxer : public IdxerBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  LocalIdxer() {}
  LocalIdxer(IdIndexer<internal_oid_t, VID_T>&& oid_indexer,
             IdIndexer<VID_T, VID_T>&& lid_indexer)
      : oid_indexer_(std::move(oid_indexer)),
        lid_indexer_(std::move(lid_indexer)) {}

  bool get_key(VID_T vid, internal_oid_t& oid) const override {
    VID_T idx;
    if (lid_indexer_.get_index(vid, idx)) {
      return oid_indexer_.get_key(idx, oid);
    } else {
      return false;
    }
  }

  bool get_index(const internal_oid_t& oid, VID_T& vid) const override {
    VID_T idx;
    if (oid_indexer_.get_index(oid, idx)) {
      return lid_indexer_.get_key(idx, vid);
    } else {
      return false;
    }
  }

  IdxerType type() const override { return IdxerType::kLocalIdxer; }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    oid_indexer_.Serialize(writer);
    lid_indexer_.Serialize(writer);
  }
  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    oid_indexer_.Deserialize(reader);
    lid_indexer_.Deserialize(reader);
  }

  size_t size() const override { return oid_indexer_.size(); }

 private:
  IdIndexer<internal_oid_t, VID_T> oid_indexer_;  // oid -> idx
  IdIndexer<VID_T, VID_T> lid_indexer_;           // lid -> idx
};

template <typename OID_T, typename VID_T>
class LocalIdxerBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override { oid_indexer_._add(oid); }

  IdxerBase<OID_T, VID_T>* finish() override {
    return new LocalIdxer<OID_T, VID_T>(std::move(oid_indexer_),
                                        std::move(lid_indexer_));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    int req_type = 1;
    sync_comm::Send(req_type, target, tag, comm_spec.comm());
    sync_comm::Send(oid_indexer_.keys(), target, tag, comm_spec.comm());
    std::vector<VID_T> response;
    sync_comm::Recv(response, target, tag + 1, comm_spec.comm());
    VID_T sentinel = std::numeric_limits<VID_T>::max();
    for (size_t i = 0; i < oid_indexer_.size(); ++i) {
      if (response[i] != std::numeric_limits<VID_T>::max()) {
        lid_indexer_._add(response[i]);
      } else {
        lid_indexer_._add(sentinel);
        --sentinel;
      }
    }
  }
  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    LOG(ERROR) << "LocalIdxerBuilder should not be used to sync response";
  }

 private:
  IdIndexer<internal_oid_t, VID_T> oid_indexer_;
  IdIndexer<VID_T, VID_T> lid_indexer_;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_LOCAL_IDXER_H_