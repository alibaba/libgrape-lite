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

#ifndef GRAPE_FRAGMENT_PARTITIONER_H_
#define GRAPE_FRAGMENT_PARTITIONER_H_

#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/config.h"

namespace grape {

/**
 * @brief HashPartitoner is a partitioner with the strategy of hashing on
 * original vertex_ids.
 *
 * @tparam OID_T
 */
template <typename OID_T>
class HashPartitioner {
 public:
  HashPartitioner() : fnum_(1) {}
  explicit HashPartitioner(size_t frag_num) : fnum_(frag_num) {}
  HashPartitioner(size_t frag_num, std::vector<OID_T>&) : fnum_(frag_num) {}

  inline fid_t GetPartitionId(const OID_T& oid) const {
    return static_cast<fid_t>(static_cast<uint64_t>(oid) % fnum_);
  }

  void SetPartitionId(const OID_T& oid, fid_t fid) {
    LOG(FATAL) << "not support";
  }

  HashPartitioner& operator=(const HashPartitioner& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    return *this;
  }

  HashPartitioner& operator=(HashPartitioner&& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    return *this;
  }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    CHECK(writer->Write(&fnum_, sizeof(fid_t)));
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    CHECK(reader->Read(&fnum_, sizeof(fid_t)));
  }

 private:
  fid_t fnum_;
};

template <>
class HashPartitioner<std::string> {
 public:
  using oid_t = std::string;
  using internal_oid_t = nonstd::string_view;

  HashPartitioner() : fnum_(1) {}
  explicit HashPartitioner(size_t frag_num) : fnum_(frag_num) {}
  HashPartitioner(size_t frag_num, std::vector<std::string>&)
      : fnum_(frag_num) {}

  inline fid_t GetPartitionId(const internal_oid_t& oid) const {
    return static_cast<fid_t>(
        static_cast<uint64_t>(std::hash<internal_oid_t>()(oid) % fnum_));
  }

  inline fid_t GetPartitionId(const oid_t& oid) const {
    internal_oid_t internal_oid(oid);
    return GetPartitionId(internal_oid);
  }

  void SetPartitionId(const oid_t& oid, fid_t fid) {
    LOG(FATAL) << "not support";
  }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) {
    LOG(FATAL) << "not support";
  }

  HashPartitioner& operator=(const HashPartitioner& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    return *this;
  }

  HashPartitioner(const HashPartitioner& other) { fnum_ = other.fnum_; }

  HashPartitioner& operator=(HashPartitioner&& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    return *this;
  }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    CHECK(writer->Write(&fnum_, sizeof(fid_t)));
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    CHECK(reader->Read(&fnum_, sizeof(fid_t)));
  }

 private:
  fid_t fnum_;
};

/**
 * @brief SegmentedPartitioner is a partitioner with a strategy of chunking
 * original vertex_ids.
 *
 * @tparam OID_T
 */
template <typename OID_T>
class SegmentedPartitioner {
 public:
  SegmentedPartitioner() : fnum_(1) {}
  explicit SegmentedPartitioner(size_t frag_num) : fnum_(frag_num) {}
  SegmentedPartitioner(size_t frag_num, std::vector<OID_T>& oid_list) {
    fnum_ = frag_num;
    size_t vnum = oid_list.size();
    size_t frag_vnum = (vnum + fnum_ - 1) / fnum_;
    o2f_.reserve(vnum);
    for (size_t i = 0; i < vnum; ++i) {
      fid_t fid = static_cast<fid_t>(i / frag_vnum);
      o2f_.emplace(oid_list[i], fid);
    }
  }

  inline fid_t GetPartitionId(const OID_T& oid) const { return o2f_.at(oid); }

  void SetPartitionId(const OID_T& oid, fid_t fid) { o2f_[oid] = fid; }

  SegmentedPartitioner& operator=(const SegmentedPartitioner& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = other.o2f_;
    return *this;
  }

  SegmentedPartitioner& operator=(SegmentedPartitioner&& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = std::move(other.o2f_);
    return *this;
  }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    InArchive arc;
    arc << fnum_ << o2f_;
    CHECK(writer->WriteArchive(arc));
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> fnum_ >> o2f_;
  }

 private:
  fid_t fnum_;
  ska::flat_hash_map<OID_T, fid_t> o2f_;
};

template <>
class SegmentedPartitioner<std::string> {
  using oid_t = std::string;
  using internal_oid_t = nonstd::string_view;

 public:
  SegmentedPartitioner() : fnum_(1) {}
  explicit SegmentedPartitioner(size_t frag_num) : fnum_(frag_num) {}
  SegmentedPartitioner(size_t frag_num, std::vector<oid_t>& oid_list) {
    fnum_ = frag_num;
    size_t vnum = oid_list.size();
    size_t frag_vnum = (vnum + fnum_ - 1) / fnum_;
    o2f_.reserve(vnum);
    for (size_t i = 0; i < vnum; ++i) {
      fid_t fid = static_cast<fid_t>(i / frag_vnum);
      o2f_.emplace(oid_list[i], fid);
    }
  }

  inline fid_t GetPartitionId(const oid_t& oid) const { return o2f_.at(oid); }

  inline fid_t GetPartitionId(const internal_oid_t& oid) const {
    return o2f_.at(std::string(oid));
  }

  void SetPartitionId(const oid_t& oid, fid_t fid) { o2f_[oid] = fid; }

  void SetPartitionId(const internal_oid_t& oid, fid_t fid) {
    o2f_[std::string(oid)] = fid;
  }

  SegmentedPartitioner& operator=(const SegmentedPartitioner& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = other.o2f_;
    return *this;
  }

  SegmentedPartitioner& operator=(SegmentedPartitioner&& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = std::move(other.o2f_);
    return *this;
  }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    InArchive arc;
    arc << fnum_ << o2f_;
    CHECK(writer->WriteArchive(arc));
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> fnum_ >> o2f_;
  }

 private:
  fid_t fnum_;
  ska::flat_hash_map<oid_t, fid_t> o2f_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_PARTITIONER_H_
