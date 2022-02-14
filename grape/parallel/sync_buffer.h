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

#ifndef GRAPE_PARALLEL_SYNC_BUFFER_H_
#define GRAPE_PARALLEL_SYNC_BUFFER_H_

#include <typeinfo>
#include <utility>

#include "grape/utils/vertex_array.h"

namespace grape {
/**
 * @brief ISyncBuffer is a base class of SyncBuffer, which is used for auto
 * parallelization.
 *
 */
class ISyncBuffer {
 public:
  virtual ~ISyncBuffer() = default;

  virtual void* data() = 0;

  virtual bool updated(size_t begin, size_t length) const = 0;

  virtual const std::type_info& GetTypeId() const = 0;

  template <typename T>
  T* base() {
    return reinterpret_cast<T*>(data());
  }
};

/**
 * @brief SyncBuffer manages status on each vertex during the evaluation in auto
 * parallization.
 *
 * @tparam VERTEX_SET_T
 * @tparam T
 */
template <typename VERTEX_SET_T, typename T>
class SyncBuffer : public ISyncBuffer {
  using vertex_t = typename VERTEX_SET_T::vertex_t;

 public:
  SyncBuffer() : data_(internal_data_) {}

  explicit SyncBuffer(VertexArray<VERTEX_SET_T, T>& data) : data_(data) {}

  bool updated(size_t begin, size_t length) const override {
    auto iter = range_.begin() + begin;
    while (length-- && iter != range_.end()) {
      if (updated_[*iter++]) {
        return true;
      }
    }
    return false;
  }

  void* data() override {
    return reinterpret_cast<void*>(&data_[*range_.begin()]);
  }

  inline const std::type_info& GetTypeId() const override { return typeid(T); }

  void Init(const VERTEX_SET_T& range, const T& value,
            const std::function<bool(T*, T&&)>& aggregator) {
    range_ = range;
    data_.Init(range, value);
    updated_.Init(range, false);
    aggregator_ = aggregator;
  }

  void SetValue(const vertex_t& v, const T& value) {
    if (value != data_[v]) {
      data_[v] = value;
      updated_[v] = true;
    }
  }

  T& GetValue(const vertex_t& v) { return data_[v]; }

  bool IsUpdated(const vertex_t& v) { return updated_[v]; }

  void SetUpdated(const vertex_t& v) { updated_[v] = true; }

  void Reset(const vertex_t& v) { updated_[v] = false; }

  void Reset(const VERTEX_SET_T& range) {
    for (auto v : range) {
      updated_[v] = false;
    }
  }

  T& operator[](const vertex_t& v) { return data_[v]; }

  const T& operator[](const vertex_t& v) const { return data_[v]; }

  void Swap(SyncBuffer& rhs) {
    data_.swap(rhs.data_);
    updated_.swap(rhs.updated_);
    range_.Swap(rhs.range_);
  }

  void Aggregate(const vertex_t v, T&& rhs) {
    bool updated = aggregator_(&data_[v], std::move(rhs));
    updated_[v] = updated_[v] || updated;
  }

 private:
  VertexArray<VERTEX_SET_T, T> internal_data_;
  VertexArray<VERTEX_SET_T, T>& data_;
  VertexArray<VERTEX_SET_T, bool> updated_;
  VERTEX_SET_T range_;

  std::function<bool(T*, T&&)> aggregator_;
};
}  // namespace grape

#endif  // GRAPE_PARALLEL_SYNC_BUFFER_H_
