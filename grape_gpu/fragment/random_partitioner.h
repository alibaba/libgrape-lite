
#ifndef GRAPE_GPU_FRAGMENT_RANDOM_PARTITIONER_H_
#define GRAPE_GPU_FRAGMENT_RANDOM_PARTITIONER_H_
#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>
#include <vector>

#include "grape_gpu/config.h"
DECLARE_int32(avg_degree);

namespace grape_gpu {
template <typename OID_T>
class RandomPartitioner {
 public:
  RandomPartitioner() : fnum_(1) {}

  RandomPartitioner(size_t frag_num, const std::vector<OID_T>& oid_list)
      : fnum_(frag_num) {
    std::vector<OID_T> shuffled_oid_list(oid_list);
    std::default_random_engine e(frag_num * oid_list.size());
    std::shuffle(shuffled_oid_list.begin(), shuffled_oid_list.end(), e);
    size_t vnum = shuffled_oid_list.size();
    size_t frag_vnum = (vnum + fnum_ - 1) / fnum_;

    o2f_.reserve(vnum);

    for (size_t i = 0; i < vnum; ++i) {
      auto fid = static_cast<fid_t>(i / frag_vnum);
      o2f_.emplace(shuffled_oid_list[i], fid);
    }
  }

  inline fid_t GetPartitionId(const OID_T& oid) { return o2f_.at(oid); }

  RandomPartitioner& operator=(const RandomPartitioner& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = other.o2f_;
    return *this;
  }

  RandomPartitioner& operator=(RandomPartitioner&& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = std::move(other.o2f_);
    return *this;
  }

 private:
  fid_t fnum_;
  std::unordered_map<OID_T, fid_t> o2f_;
};

/**
 * @brief SegmentedPartitioner is a partitioner with a strategy of chunking
 * original vertex_ids.
 *
 * @tparam OID_T
 */
template <typename OID_T>
class SegHashPartitioner {
 public:
  SegHashPartitioner() : fnum_(1) {}

  SegHashPartitioner(size_t frag_num, std::vector<OID_T>& oid_list) {
    fnum_ = frag_num;
    size_t vnum = oid_list.size();
    size_t chunk_size = FLAGS_avg_degree;

    o2f_.reserve(vnum);
    for (size_t i = 0; i < vnum; ++i) {
      auto fid = static_cast<fid_t>(i / chunk_size % frag_num);
      o2f_.emplace(oid_list[i], fid);
    }
  }

  inline fid_t GetPartitionId(const OID_T& oid) { return o2f_.at(oid); }

  SegHashPartitioner& operator=(const SegHashPartitioner& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = other.o2f_;
    return *this;
  }

  SegHashPartitioner& operator=(SegHashPartitioner&& other) {
    if (this == &other) {
      return *this;
    }
    fnum_ = other.fnum_;
    o2f_ = std::move(other.o2f_);
    return *this;
  }

 private:
  fid_t fnum_;
  ska::flat_hash_map<OID_T, fid_t> o2f_;
};
}  // namespace grape_gpu

#endif  // GRAPE_GPU_FRAGMENT_RANDOM_PARTITIONER_H_
