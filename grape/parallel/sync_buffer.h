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
 * @tparam T
 * @tparam VID_T
 */
template <typename T, typename VID_T>
class SyncBuffer : public ISyncBuffer {
 public:
  SyncBuffer() : data_(internal_data_) {}

  explicit SyncBuffer(VertexArray<T, VID_T>& data) : data_(data) {}

  bool updated(size_t begin, size_t length) const override {
    auto iter = updated_.begin() + begin;
    auto end = updated_.begin() + length;
    for (; iter != end; ++iter) {
      if (*iter) {
        return true;
      }
    }
    return false;
  }

  void* data() override {
    return reinterpret_cast<void*>(&data_[range_.begin()]);
  }

  inline const std::type_info& GetTypeId() const override { return typeid(T); }

  void Init(VertexRange<VID_T> range, const T& value,
            const std::function<bool(T*, T&&)>& aggregator) {
    range_ = range;
    data_.Init(range, value);
    updated_.Init(range, false);
    aggregator_ = aggregator;
  }

  void SetValue(Vertex<VID_T> v, const T& value) {
    if (value != data_[v]) {
      data_[v] = value;
      updated_[v] = true;
    }
  }

  T& GetValue(Vertex<VID_T> v) { return data_[v]; }

  bool IsUpdated(Vertex<VID_T> v) { return updated_[v]; }

  void SetUpdated(Vertex<VID_T> v) { updated_[v] = true; }

  void Reset(Vertex<VID_T> v) { updated_[v] = false; }

  void Reset(VertexRange<VID_T> range) {
    for (auto v : range) {
      updated_[v] = false;
    }
  }

  T& operator[](Vertex<VID_T> v) { return data_[v]; }

  const T& operator[](Vertex<VID_T> v) const { return data_[v]; }

  void Swap(SyncBuffer<T, VID_T>& rhs) {
    data_.swap(rhs.data_);
    updated_.swap(rhs.updated_);
    range_.Swap(rhs.range_);
  }

  void Aggregate(Vertex<VID_T> v, T&& rhs) {
    bool updated = aggregator_(&data_[v], std::move(rhs));
    updated_[v] = updated_[v] || updated;
  }

 private:
  VertexArray<T, VID_T> internal_data_;
  VertexArray<T, VID_T>& data_;
  VertexArray<bool, VID_T> updated_;
  VertexRange<VID_T> range_;

  std::function<bool(T*, T&&)> aggregator_;
};
}  // namespace grape

#endif  // GRAPE_PARALLEL_SYNC_BUFFER_H_
