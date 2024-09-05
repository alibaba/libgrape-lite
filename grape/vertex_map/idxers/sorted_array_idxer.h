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

#ifndef GRAPE_VERTEX_MAP_IDXERS_SORTED_ARRAY_IDXER_H_
#define GRAPE_VERTEX_MAP_IDXERS_SORTED_ARRAY_IDXER_H_

#include "grape/utils/gcontainer.h"
#include "grape/vertex_map/idxers/idxer_base.h"

namespace grape {

template <typename OID_T, typename VID_T>
class SortedArrayIdxer : public IdxerBase<OID_T, VID_T> {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  SortedArrayIdxer() {}
  explicit SortedArrayIdxer(Array<OID_T, Allocator<OID_T>>&& id_list)
      : id_list_(std::move(id_list)) {}
  ~SortedArrayIdxer() {}

  bool get_key(VID_T vid, internal_oid_t& oid) const override {
    if (vid >= id_list_.size()) {
      return false;
    }
    oid = id_list_[vid];
    return true;
  }

  bool get_index(const internal_oid_t& oid, VID_T& vid) const override {
    auto it = std::lower_bound(id_list_.begin(), id_list_.end(), oid);
    if (it == id_list_.end() || *it != oid) {
      return false;
    }
    vid = it - id_list_.begin();
    return true;
  }

  IdxerType type() const override { return IdxerType::kSortedArrayIdxer; }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    size_t size = id_list_.size();
    writer->Write(&size, sizeof(size_t));
    writer->Write(id_list_.data(), size * sizeof(OID_T));
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    size_t size;
    reader->Read(&size, sizeof(size_t));
    id_list_.resize(size);
    reader->Read(id_list_.data(), size * sizeof(OID_T));
  }

  size_t size() const override { return id_list_.size(); }

  size_t memory_usage() const override {
    return id_list_.size() * sizeof(OID_T);
  }

 private:
  Array<OID_T, Allocator<OID_T>> id_list_;
};

template <typename VID_T>
class SortedArrayIdxer<std::string, VID_T>
    : public IdxerBase<std::string, VID_T> {
  using internal_oid_t = typename InternalOID<std::string>::type;

 public:
  SortedArrayIdxer() {}
  explicit SortedArrayIdxer(StringViewVector&& id_list)
      : id_list_(std::move(id_list)) {}
  ~SortedArrayIdxer() {}

  bool get_key(VID_T vid, internal_oid_t& oid) const override {
    if (vid >= id_list_.size()) {
      return false;
    }
    oid = internal_oid_t(id_list_[vid]);
    return true;
  }

  bool get_index(const internal_oid_t& oid, VID_T& vid) const override {
    size_t num = id_list_.size();
    size_t low = 0, high = num - 1;
    nonstd::string_view oid_view(oid);
    while (low <= high) {
      size_t mid = low + (high - low) / 2;
      if (id_list_[mid] == oid_view) {
        vid = mid;
        return true;
      } else if (id_list_[mid] < oid_view) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return false;
  }

  IdxerType type() const override { return IdxerType::kSortedArrayIdxer; }

  void serialize(std::unique_ptr<IOAdaptorBase>& writer) override {
    id_list_.serialize(writer);
  }

  void deserialize(std::unique_ptr<IOAdaptorBase>& reader) override {
    id_list_.deserialize(reader);
  }

  size_t size() const override { return id_list_.size(); }

  size_t memory_usage() const override {
    return id_list_.content_buffer().size() +
           id_list_.offset_buffer().size() * sizeof(size_t);
  }

 private:
  StringViewVector id_list_;
};

template <typename OID_T, typename VID_T>
class SortedArrayIdxerDummyBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override {}

  std::unique_ptr<IdxerBase<OID_T, VID_T>> finish() override {
    return std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new SortedArrayIdxer<OID_T, VID_T>(std::move(id_list_)));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    sync_comm::Recv(id_list_, target, tag, comm_spec.comm());
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    LOG(ERROR) << "SortedArrayIdxerDummyBuilder should not be used to sync "
                  "response";
  }

 private:
  Array<OID_T, Allocator<OID_T>> id_list_;
};

template <typename VID_T>
class SortedArrayIdxerDummyBuilder<std::string, VID_T>
    : public IdxerBuilderBase<std::string, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<std::string>::type;
  void add(const internal_oid_t& oid) override {}

  std::unique_ptr<IdxerBase<std::string, VID_T>> finish() override {
    return std::unique_ptr<IdxerBase<std::string, VID_T>>(
        new SortedArrayIdxer<std::string, VID_T>(std::move(id_list_)));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    sync_comm::Recv(id_list_, target, tag, comm_spec.comm());
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    LOG(ERROR) << "SortedArrayIdxerDummyBuilder should not be used to sync "
                  "response";
  }

 private:
  StringViewVector id_list_;
};

template <typename OID_T, typename VID_T>
class SortedArrayIdxerBuilder : public IdxerBuilderBase<OID_T, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<OID_T>::type;
  void add(const internal_oid_t& oid) override { keys_.push_back(OID_T(oid)); }

  std::unique_ptr<IdxerBase<OID_T, VID_T>> finish() override {
    if (!sorted_) {
      DistinctSort(keys_);
      sorted_ = true;
    }
    Array<OID_T, Allocator<OID_T>> id_list(keys_.size());
    std::copy(keys_.begin(), keys_.end(), id_list.begin());
    return std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new SortedArrayIdxer<OID_T, VID_T>(std::move(id_list)));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    LOG(ERROR) << "HashMapIdxerBuilder should not be used to sync request";
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    if (!sorted_) {
      DistinctSort(keys_);
      sorted_ = true;
    }
    sync_comm::Send(keys_, source, tag, comm_spec.comm());
  }

 private:
  std::vector<OID_T> keys_;
  bool sorted_ = false;
};

template <typename VID_T>
class SortedArrayIdxerBuilder<std::string, VID_T>
    : public IdxerBuilderBase<std::string, VID_T> {
 public:
  using internal_oid_t = typename InternalOID<std::string>::type;
  void add(const internal_oid_t& oid) override {
    keys_.push_back(std::string(oid));
  }

  std::unique_ptr<IdxerBase<std::string, VID_T>> finish() override {
    if (!sorted_) {
      DistinctSort(keys_);
      for (auto& key : keys_) {
        id_list_.emplace_back(key);
      }
      sorted_ = true;
    }
    return std::unique_ptr<IdxerBase<std::string, VID_T>>(
        new SortedArrayIdxer<std::string, VID_T>(std::move(id_list_)));
  }

  void sync_request(const CommSpec& comm_spec, int target, int tag) override {
    LOG(ERROR) << "HashMapIdxerBuilder should not be used to sync request";
  }

  void sync_response(const CommSpec& comm_spec, int source, int tag) override {
    if (!sorted_) {
      DistinctSort(keys_);
      for (auto& key : keys_) {
        id_list_.emplace_back(key);
      }
      sorted_ = true;
    }
    sync_comm::Send(id_list_, source, tag, comm_spec.comm());
  }

 private:
  std::vector<std::string> keys_;
  StringViewVector id_list_;
  bool sorted_ = false;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_SORTED_ARRAY_IDXER_H_
