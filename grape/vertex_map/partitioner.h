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
  kSegmentedPartitioner,
};

inline PartitionerType parse_partitioner_type_name(const std::string& name) {
  if (name == "hash") {
    return PartitionerType::kHashPartitioner;
  } else if (name == "map") {
    return PartitionerType::kMapPartitioner;
  } else if (name == "segment") {
    return PartitionerType::kSegmentedPartitioner;
  } else {
    LOG(ERROR) << "unrecognized partitioner: " << name
               << ", use map partitioner as default";
    return PartitionerType::kMapPartitioner;
  }
}

template <typename OID_T>
class IPartitioner {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;

  virtual ~IPartitioner() = default;

  virtual fid_t GetPartitionId(const internal_oid_t& oid) const = 0;

  virtual void SetPartitionId(const internal_oid_t& oid, fid_t fid) = 0;

  virtual void serialize(std::unique_ptr<IOAdaptorBase>& writer) = 0;

  virtual void deserialize(std::unique_ptr<IOAdaptorBase>& reader) = 0;

  virtual size_t memory_usage() const = 0;

  virtual PartitionerType type() const = 0;
};

template <typename OID_T, typename HASH_T = std::hash<OID_T>>
class HashPartitioner : public IPartitioner<OID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;

  HashPartitioner() : hash_(), fnum_(1) {}
  explicit HashPartitioner(size_t frag_num) : hash_(), fnum_(frag_num) {}

  fid_t GetPartitionId(const internal_oid_t& oid) const override {
    return static_cast<fid_t>(hash_(OID_T(oid)) % fnum_);
  }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) override {
    if (GetPartitionId(oid) != fid) {
      LOG(ERROR) << "HashPartitioner cannot set partition id";
    }
  }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    CHECK(writer->Write(&fnum_, sizeof(fid_t)));
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    CHECK(reader->Read(&fnum_, sizeof(fid_t)));
  }

  PartitionerType type() const override {
    return PartitionerType::kHashPartitioner;
  }

  size_t memory_usage() const override { return 0; }

 private:
  HASH_T hash_;
  fid_t fnum_;
};

template <typename OID_T>
class MapPartitioner : public IPartitioner<OID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;

  MapPartitioner() : fnum_(0) {}
  explicit MapPartitioner(fid_t fnum) : fnum_(fnum) {}
  MapPartitioner(fid_t fnum, const std::vector<OID_T>& oid_list) {
    fnum_ = fnum;
    Init(fnum, oid_list);
  }
  ~MapPartitioner() = default;

  void Init(fid_t fnum, const std::vector<OID_T>& oid_list) {
    fnum_ = fnum;
    size_t frag_num = fnum;
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
    fnum_ = frag_num;
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
      return fnum_;
    }
    return iter->second;
  }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) override {
    o2f_[OID_T(oid)] = fid;
  }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    InArchive arc;
    arc << fnum_ << o2f_;
    CHECK(writer->WriteArchive(arc));
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> fnum_ >> o2f_;
  }

  PartitionerType type() const override {
    return PartitionerType::kMapPartitioner;
  }

  size_t memory_usage() const override { return o2f_.memory_usage(); }

 private:
  fid_t fnum_;
  ska::flat_hash_map<OID_T, fid_t> o2f_;
};

template <typename OID_T>
class SegmentedPartitioner : public IPartitioner<OID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  SegmentedPartitioner() : fnum_(0) {}
  SegmentedPartitioner(fid_t fnum, const std::vector<OID_T>& sorted_oid_list) {
    fnum_ = fnum;
    size_t part_size = (sorted_oid_list.size() + fnum - 1) / fnum;
    for (size_t i = 1; i < fnum; ++i) {
      boundaries_.emplace_back(sorted_oid_list[i * part_size]);
    }
  }
  explicit SegmentedPartitioner(const std::vector<OID_T>& boundaries)
      : fnum_(boundaries.size() + 1), boundaries_(boundaries) {}
  ~SegmentedPartitioner() = default;

  void Init(fid_t fnum, const std::vector<OID_T>& boundaries) {
    fnum_ = fnum;
    boundaries_ = boundaries;
    CHECK_EQ(fnum_, boundaries_.size() + 1);
  }

  fid_t GetPartitionId(const internal_oid_t& oid) const override {
    auto iter =
        std::upper_bound(boundaries_.begin(), boundaries_.end(), OID_T(oid));
    return static_cast<fid_t>(iter - boundaries_.begin());
  }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) override {
    LOG(FATAL) << "SegmentedPartitioner cannot set partition id";
  }

  PartitionerType type() const override {
    return PartitionerType::kSegmentedPartitioner;
  }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    InArchive arc;
    arc << fnum_ << boundaries_;
    CHECK(writer->WriteArchive(arc));
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> fnum_ >> boundaries_;
  }

  size_t memory_usage() const override {
    return boundaries_.size() * sizeof(OID_T);
  }

 private:
  fid_t fnum_;
  std::vector<OID_T> boundaries_;
};

template <typename OID_T>
void serialize_partitioner(std::unique_ptr<IOAdaptorBase>& writer,
                           std::unique_ptr<IPartitioner<OID_T>>& partitioner) {
  int type = static_cast<int>(partitioner->type());
  writer->Write(&type, sizeof(type));
  partitioner->serialize(writer);
}

template <typename OID_T>
std::unique_ptr<IPartitioner<OID_T>> deserialize_partitioner(
    std::unique_ptr<IOAdaptorBase>& reader) {
  int type;
  reader->Read(&type, sizeof(type));
  std::unique_ptr<IPartitioner<OID_T>> partitioner(nullptr);
  switch (static_cast<PartitionerType>(type)) {
  case PartitionerType::kHashPartitioner:
    partitioner =
        std::unique_ptr<IPartitioner<OID_T>>(new HashPartitioner<OID_T>());
    break;
  case PartitionerType::kMapPartitioner:
    partitioner =
        std::unique_ptr<IPartitioner<OID_T>>(new MapPartitioner<OID_T>());
    break;
  case PartitionerType::kSegmentedPartitioner:
    partitioner =
        std::unique_ptr<IPartitioner<OID_T>>(new SegmentedPartitioner<OID_T>());
    break;
  default:
    LOG(FATAL) << "Unknown partitioner type";
  }
  if (partitioner) {
    partitioner->deserialize(reader);
  }
  return partitioner;
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_PARTITIONER_H_
