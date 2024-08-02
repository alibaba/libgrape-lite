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

#ifndef GRAPE_VERTEX_MAP_PARTITIONER_H_
#define GRAPE_VERTEX_MAP_PARTITIONER_H_

#include <string>
#include "grape/io/io_adaptor_base.h"

namespace grape {

enum class PartitionerType {
  kHashPartitioner,
  kMapPartitioner,
};

template <typename OID_T>
class IPartitioner {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;

  virtual ~IPartitioner() = default;

  virtual fid_t GetPartitionId(const internal_oid_t& oid) const = 0;

  virtual void SetPartitionId(const internal_oid_t& oid, fid_t fid) = 0;

  virtual void serialize(IOAdaptorBase* writer) = 0;

  virtual void deserialize(IOAdaptorBase* reader) = 0;

  virtual PartitionerType type() const = 0;
};

template <typename OID_T, typename HASH_T = std::hash<OID_T>>
class HashPartitionerBeta : public IPartitioner<OID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;

  HashPartitionerBeta() : hash_(), fnum_(1) {}
  explicit HashPartitionerBeta(size_t frag_num) : hash_(), fnum_(frag_num) {}

  fid_t GetPartitionId(const internal_oid_t& oid) const override {
    return static_cast<fid_t>(hash_(OID_T(oid)) % fnum_);
  }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) override {
    if (GetPartitionId(oid) != fid) {
      LOG(ERROR) << "HashPartitioner cannot set partition id";
    }
  }

  void serialize(IOAdaptorBase* writer) override {
    CHECK(writer->Write(&fnum_, sizeof(fid_t)));
  }

  void deserialize(IOAdaptorBase* reader) override {
    CHECK(reader->Read(&fnum_, sizeof(fid_t)));
  }

  PartitionerType type() const override {
    return PartitionerType::kHashPartitioner;
  }

 private:
  HASH_T hash_;
  fid_t fnum_;
};

template <typename OID_T>
class MapPartitioner : public IPartitioner<OID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;

  MapPartitioner() = default;
  MapPartitioner(size_t fnum, const std::vector<OID_T>& oid_list) {
    Init(fnum, oid_list);
  }
  ~MapPartitioner() = default;

  void Init(size_t frag_num, const std::vector<OID_T>& oid_list) {
    size_t vnum = oid_list.size();
    size_t frag_vnum = (vnum + frag_num - 1) / frag_num;
    o2f_.clear();
    o2f_.reserve(vnum);
    for (size_t i = 0; i < vnum; ++i) {
      fid_t fid = static_cast<fid_t>(i / frag_vnum);
      o2f_.emplace(oid_list[i], fid);
    }
  }

  void Init(const std::vector<std::vector<OID_T>>& oid_lists) {
    size_t frag_num = oid_lists.size();
    o2f_.clear();
    for (size_t i = 0; i < frag_num; ++i) {
      for (const auto& oid : oid_lists[i]) {
        o2f_.emplace(oid, i);
      }
    }
  }

  fid_t GetPartitionId(const internal_oid_t& oid) const override {
    auto iter = o2f_.find(OID_T(oid));
    if (iter == o2f_.end()) {
      return -1;
    }
    return iter->second;
  }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) override {
    o2f_[OID_T(oid)] = fid;
  }

  void serialize(IOAdaptorBase* writer) override {
    InArchive arc;
    arc << o2f_;
    CHECK(writer->WriteArchive(arc));
  }

  void deserialize(IOAdaptorBase* reader) override {
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> o2f_;
  }

  PartitionerType type() const override {
    return PartitionerType::kMapPartitioner;
  }

 private:
  ska::flat_hash_map<OID_T, fid_t> o2f_;
};

template <typename OID_T>
void serialize_partitioner(IOAdaptorBase* writer,
                           IPartitioner<OID_T>* partitioner) {
  int type = static_cast<int>(partitioner->type());
  writer->Write(&type, sizeof(type));
  partitioner->serialize(writer);
}

template <typename OID_T>
IPartitioner<OID_T>* deserialize_partitioner(IOAdaptorBase* reader) {
  int type;
  reader->Read(&type, sizeof(type));
  IPartitioner<OID_T>* partitioner = nullptr;
  switch (static_cast<PartitionerType>(type)) {
  case PartitionerType::kHashPartitioner:
    partitioner = new HashPartitionerBeta<OID_T>();
    break;
  case PartitionerType::kMapPartitioner:
    partitioner = new MapPartitioner<OID_T>();
    break;
  default:
    LOG(FATAL) << "Unknown partitioner type";
  }
  partitioner->deserialize(reader);
  return partitioner;
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_PARTITIONER_H_