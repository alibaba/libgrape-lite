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

#ifndef GRAPE_VERTEX_MAP_IDXER_H_
#define GRAPE_VERTEX_MAP_IDXER_H_

#include "grape/communication/sync_comm.h"
#include "grape/graph/id_indexer.h"
#include "grape/io/io_adaptor_base.h"
#include "grape/worker/comm_spec.h"

namespace grape {

enum class IdxerType {
  kHashMapIdxer,
  kLocalIdxer,
  kPerfectHashIdxer,
};

template <typename OID_T, typename VID_T>
class IdxerBase {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  virtual ~IdxerBase() = default;

  virtual bool get_key(VID_T vid, internal_oid_t& oid) const = 0;

  virtual bool get_index(const internal_oid_t& oid, VID_T& vid) const = 0;

  virtual IdxerType type() const = 0;

  virtual size_t size() const = 0;

  virtual void serialize(IOAdaptorBase* writer) = 0;
  virtual void deserialize(IOAdaptorBase* reader) = 0;
};

template <typename OID_T, typename VID_T>
class IdxerBuilderBase {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  virtual ~IdxerBuilderBase() = default;

  virtual void add(const internal_oid_t& oid) = 0;

  virtual IdxerBase<OID_T, VID_T>* finish() = 0;

  virtual void sync_request(const CommSpec& comm_spec, int target, int tag) = 0;
  virtual void sync_response(const CommSpec& comm_spec, int source,
                             int tag) = 0;
};

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

  void serialize(IOAdaptorBase* writer) override {
    oid_indexer_.Serialize(writer);
    lid_indexer_.Serialize(writer);
  }
  void deserialize(IOAdaptorBase* reader) override {
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

template <typename OID_T, typename VID_T>
class DummyIdxerBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override {}

  IdxerBase<OID_T, VID_T>* finish() override {
    return new HashMapIdxer<OID_T, VID_T>(std::move(indexer_));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    int req_type = 0;
    sync_comm::Send(req_type, target, tag, comm_spec.comm());
    sync_comm::Recv(indexer_, target, tag + 1, comm_spec.comm());
  }
  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    LOG(ERROR) << "DummyIdxerBuilder should not be used to sync response";
  }

 private:
  IdIndexer<internal_oid_t, VID_T> indexer_;
};

template <typename OID_T, typename VID_T>
class PerfectHashIdxer : public IdxerBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  PerfectHashIdxer() {}
  PerfectHashIdxer(IdIndexer<internal_oid_t, VID_T>&& oid_indexer,
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

  IdxerType type() const override { return IdxerType::kPerfectHashIdxer; }

  void serialize(IOAdaptorBase* writer) override {
    oid_indexer_.Serialize(writer);
    lid_indexer_.Serialize(writer);
  }
  void deserialize(IOAdaptorBase* reader) override {
    oid_indexer_.Deserialize(reader);
    lid_indexer_.Deserialize(reader);
  }

  size_t size() const override { return oid_indexer_.size(); }

 private:
  IdIndexer<internal_oid_t, VID_T> oid_indexer_;  // oid -> idx
  IdIndexer<VID_T, VID_T> lid_indexer_;           // lid -> idx
};

template <typename OID_T, typename VID_T>
void serialize_idxer(IOAdaptorBase* writer, IdxerBase<OID_T, VID_T>* idxer) {
  int type = static_cast<int>(idxer->type());
  writer->Write(&type, sizeof(type));
  idxer->serialize(writer);
}

template <typename OID_T, typename VID_T>
IdxerBase<OID_T, VID_T>* deserialize_idxer(IOAdaptorBase* reader) {
  int type;
  reader->Read(&type, sizeof(type));
  IdxerType idxer_type = static_cast<IdxerType>(type);
  switch (idxer_type) {
  case IdxerType::kHashMapIdxer: {
    auto idxer = new HashMapIdxer<OID_T, VID_T>();
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kLocalIdxer: {
    auto idxer = new LocalIdxer<OID_T, VID_T>();
    idxer->deserialize(reader);
    return idxer;
  }
  default:
    return nullptr;
  }
}

template <typename OID_T, typename VID_T>
IdxerBase<OID_T, VID_T>* extend_indexer(IdxerBase<OID_T, VID_T>* input,
                                        const std::vector<OID_T>& id_list) {
  if (input->type() == IdxerType::kHashMapIdxer) {
    auto casted = dynamic_cast<HashMapIdxer<OID_T, VID_T>*>(input);
    for (auto& id : id_list) {
      casted->add(id);
    }
    return input;
  } else {
    LOG(FATAL) << "Only HashMapIdxer can be extended";
  }
  return nullptr;
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXER_H_
