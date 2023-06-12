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

#ifndef GRAPE_SERIALIZATION_IN_ARCHIVE_H_
#define GRAPE_SERIALIZATION_IN_ARCHIVE_H_

#include <string.h>

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/types.h"
#include "grape/utils/gcontainer.h"

namespace grape {

class OutArchive;

/**
 * @brief InArchive is an archived object for serializing objects.
 *
 */
class InArchive {
 public:
  InArchive() {}
  InArchive(InArchive&& rhs) { buffer_.swap(rhs.buffer_); }

  ~InArchive() {}

  InArchive& operator=(InArchive&& rhs) {
    buffer_.clear();
    buffer_.swap(rhs.buffer_);
    return *this;
  }

  inline void Reset() { buffer_.clear(); }

  inline char* GetBuffer() { return buffer_.data(); }

  inline const char* GetBuffer() const { return buffer_.data(); }

  inline size_t GetSize() const { return buffer_.size(); }

  inline void AddByte(char v) { buffer_.push_back(v); }

  inline void AddBytes(const void* head, size_t size) {
    size_t _size = buffer_.size();
    buffer_.resize(_size + size);
    memcpy(&buffer_[_size], head, size);
  }

  inline void Resize(size_t size) { buffer_.resize(size); }

  inline void Clear() { buffer_.clear(); }

  friend class OutArchive;

  bool Empty() const { return buffer_.empty(); }

  void Reserve(size_t cap) { buffer_.reserve(cap); }

 private:
  std::vector<char> buffer_;
};

template <typename T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline InArchive& operator<<(InArchive& in_archive, T u) {
  in_archive.AddBytes(&u, sizeof(T));
  return in_archive;
}

inline InArchive& operator<<(InArchive& in_archive, EmptyType) {
  return in_archive;
}

inline InArchive& operator<<(InArchive& in_archive, const std::string& str) {
  size_t size = str.size();
  in_archive << size;
  in_archive.AddBytes(str.data(), size);
  return in_archive;
}

inline InArchive& operator<<(InArchive& archive,
                             const nonstd::string_view& str) {
  archive << str.length();
  archive.AddBytes(str.data(), str.length());
  return archive;
}

template <typename T1, typename T2>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::pair<T1, T2>& p) {
  in_archive << p.first;
  in_archive << p.second;
  return in_archive;
}

template <typename T1, typename T2, typename T3>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::tuple<T1, T2, T3>& t) {
  in_archive << std::get<0>(t);
  in_archive << std::get<1>(t);
  in_archive << std::get<2>(t);

  return in_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::vector<T, ALLOC_T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  in_archive.AddBytes(vec.data(), size * sizeof(T));
  return in_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<!std::is_pod<T>::value, T>::type* = nullptr>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::vector<T, ALLOC_T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  for (auto& item : vec) {
    in_archive << item;
  }
  return in_archive;
}

template <typename ALLOC_T>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::vector<EmptyType, ALLOC_T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  return in_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline InArchive& operator<<(InArchive& in_archive,
                             const Array<T, ALLOC_T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  in_archive.AddBytes(vec.data(), size * sizeof(T));
  return in_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<!std::is_pod<T>::value, T>::type* = nullptr>
inline InArchive& operator<<(InArchive& in_archive,
                             const Array<T, ALLOC_T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  for (auto& item : vec) {
    in_archive << item;
  }
  return in_archive;
}

template <typename ALLOC_T>
inline InArchive& operator<<(InArchive& in_archive,
                             const Array<EmptyType, ALLOC_T>& vec) {
  size_t size = vec.size();
  in_archive << size;
  return in_archive;
}

template <typename T>
inline InArchive& operator<<(InArchive& in_archive, const std::set<T>& s) {
  size_t size = s.size();
  in_archive << size;
  for (auto& item : s) {
    in_archive << item;
  }
  return in_archive;
}

template <typename T>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::unordered_set<T>& s) {
  size_t size = s.size();
  in_archive << size;
  for (auto& item : s) {
    in_archive << item;
  }
  return in_archive;
}

template <typename T1, typename T2>
inline InArchive& operator<<(InArchive& in_archive, const std::map<T1, T2>& m) {
  size_t size = m.size();
  in_archive << size;
  for (auto& pair : m) {
    in_archive << pair;
  }
  return in_archive;
}

template <typename T1, typename T2>
inline InArchive& operator<<(InArchive& in_archive,
                             const std::unordered_map<T1, T2>& m) {
  size_t size = m.size();
  in_archive << size;
  for (auto& pair : m) {
    in_archive << pair;
  }
  return in_archive;
}

template <typename T1, typename T2>
inline InArchive& operator<<(InArchive& in_archive,
                             const ska::flat_hash_map<T1, T2>& m) {
  size_t size = m.size();
  in_archive << size;
  for (auto& pair : m) {
    in_archive << pair;
  }
  return in_archive;
}

}  // namespace grape

#endif  // GRAPE_SERIALIZATION_IN_ARCHIVE_H_
