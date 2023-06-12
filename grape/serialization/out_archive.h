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

#ifndef GRAPE_SERIALIZATION_OUT_ARCHIVE_H_
#define GRAPE_SERIALIZATION_OUT_ARCHIVE_H_

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
#include "grape/serialization/in_archive.h"
#include "grape/types.h"
#include "grape/utils/gcontainer.h"

namespace grape {

/**
 * @brief OutArchive is an archived object for deserializing objects.
 *
 */
class OutArchive {
 public:
  OutArchive() : begin_(NULL), end_(NULL) {}

  OutArchive(const OutArchive& rhs) : buffer_(rhs.buffer_) {
    if (buffer_.empty()) {
      if (!rhs.Empty()) {
        buffer_.resize(rhs.GetSize());
        memcpy(&buffer_[0], rhs.begin_, rhs.GetSize());
        begin_ = &buffer_[0];
        end_ = begin_ + rhs.GetSize();
      } else {
        begin_ = NULL;
        end_ = NULL;
      }
    } else {
      begin_ = rhs.begin_ - &rhs.buffer_[0] + &buffer_[0];
      end_ = rhs.end_ - &rhs.buffer_[0] + &buffer_[0];
    }
  }

  explicit OutArchive(size_t size)
      : buffer_(size), begin_(buffer_.data()), end_(begin_ + size) {}

  OutArchive(OutArchive&& oa) {
    buffer_.swap(oa.buffer_);
    begin_ = oa.begin_;
    end_ = oa.end_;
  }

  explicit OutArchive(InArchive&& ia) {
    buffer_.swap(ia.buffer_);
    begin_ = buffer_.data();
    end_ = begin_ + static_cast<ptrdiff_t>(buffer_.size());
  }

  ~OutArchive() {}

  OutArchive& operator=(InArchive&& rhs) {
    buffer_.clear();
    buffer_.swap(rhs.buffer_);
    begin_ = buffer_.data();
    end_ = begin_ + static_cast<ptrdiff_t>(buffer_.size());
    return *this;
  }

  OutArchive& operator=(OutArchive&& rhs) {
    buffer_.clear();
    buffer_.swap(rhs.buffer_);
    begin_ = rhs.begin_;
    end_ = rhs.end_;
    rhs.begin_ = NULL;
    rhs.end_ = NULL;
    return *this;
  }

  inline void Clear() {
    buffer_.clear();
    begin_ = NULL;
    end_ = NULL;
  }

  inline void Allocate(size_t size) {
    buffer_.resize(size);
    begin_ = buffer_.data();
    end_ = begin_ + static_cast<ptrdiff_t>(size);
  }

  inline void Rewind() { begin_ = buffer_.data(); }

  inline void SetSlice(char* buffer, size_t size) {
    buffer_.clear();
    begin_ = buffer;
    end_ = begin_ + size;
  }

  inline char* GetBuffer() { return begin_; }

  inline const char* GetBuffer() const { return begin_; }

  inline size_t GetSize() const { return end_ - begin_; }

  inline bool Empty() const { return (begin_ == end_); }

  inline void* GetBytes(unsigned int size) {
    char* ret = begin_;
    begin_ += size;
    return ret;
  }

  template <typename T>
  inline void Peek(T& value) {
    char* old_begin = begin_;
    *this >> value;
    begin_ = old_begin;
  }

  std::vector<char> buffer_;
  char* begin_;
  char* end_;
};

template <typename T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline OutArchive& operator>>(OutArchive& out_archive, T& u) {
  u = *reinterpret_cast<T*>(out_archive.GetBytes(sizeof(T)));
  return out_archive;
}

inline OutArchive& operator>>(OutArchive& out_archive, EmptyType&) {
  return out_archive;
}

inline OutArchive& operator>>(OutArchive& out_archive, std::string& str) {
  size_t size;
  out_archive >> size;
  str.resize(size);
  memcpy(&str[0], out_archive.GetBytes(size), size);
  return out_archive;
}

inline OutArchive& operator>>(OutArchive& archive, nonstd::string_view& str) {
  size_t length;
  archive >> length;
  str = nonstd::string_view(reinterpret_cast<char*>(archive.GetBytes(length)),
                            length);
  return archive;
}

template <typename T1, typename T2>
inline OutArchive& operator>>(OutArchive& out_archive, std::pair<T1, T2>& p) {
  out_archive >> p.first;
  out_archive >> p.second;
  return out_archive;
}

template <typename T1, typename T2, typename T3>
inline OutArchive& operator>>(OutArchive& out_archive,
                              std::tuple<T1, T2, T3>& t) {
  out_archive >> std::get<0>(t);
  out_archive >> std::get<1>(t);
  out_archive >> std::get<2>(t);

  return out_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline OutArchive& operator>>(OutArchive& out_archive,
                              std::vector<T, ALLOC_T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  memcpy(&vec[0], out_archive.GetBytes(sizeof(T) * size), sizeof(T) * size);
  return out_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<!std::is_pod<T>::value, T>::type* = nullptr>
inline OutArchive& operator>>(OutArchive& out_archive,
                              std::vector<T, ALLOC_T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  for (auto& item : vec) {
    out_archive >> item;
  }
  return out_archive;
}

template <typename ALLOC_T>
inline OutArchive& operator>>(OutArchive& out_archive,
                              std::vector<EmptyType, ALLOC_T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  return out_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline OutArchive& operator>>(OutArchive& out_archive, Array<T, ALLOC_T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  memcpy(&vec[0], out_archive.GetBytes(sizeof(T) * size), sizeof(T) * size);
  return out_archive;
}

template <typename T, typename ALLOC_T,
          typename std::enable_if<!std::is_pod<T>::value, T>::type* = nullptr>
inline OutArchive& operator>>(OutArchive& out_archive, Array<T, ALLOC_T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  for (auto& item : vec) {
    out_archive >> item;
  }
  return out_archive;
}

template <typename ALLOC_T>
inline OutArchive& operator>>(OutArchive& out_archive,
                              Array<EmptyType, ALLOC_T>& vec) {
  size_t size;
  out_archive >> size;
  vec.resize(size);
  return out_archive;
}

template <typename T>
inline OutArchive& operator>>(OutArchive& out_archive, std::set<T>& s) {
  s.clear();
  size_t size;
  out_archive >> size;
  T item;
  while (size--) {
    out_archive >> item;
    s.insert(item);
  }
  return out_archive;
}

template <typename T>
inline OutArchive& operator>>(OutArchive& out_archive,
                              std::unordered_set<T>& s) {
  s.clear();
  size_t size;
  out_archive >> size;
  T item;
  while (size--) {
    out_archive >> item;
    s.insert(item);
  }
  return out_archive;
}

template <typename T1, typename T2>
inline OutArchive& operator>>(OutArchive& out_archive, std::map<T1, T2>& m) {
  m.clear();
  size_t size;
  out_archive >> size;
  T1 key;
  T2 value;
  while (size--) {
    out_archive >> key >> value;
    m.emplace(key, value);
  }
  return out_archive;
}

template <typename T1, typename T2>
inline OutArchive& operator>>(OutArchive& out_archive,
                              std::unordered_map<T1, T2>& m) {
  m.clear();
  size_t size;
  out_archive >> size;
  T1 key;
  T2 value;
  while (size--) {
    out_archive >> key >> value;
    m.emplace(key, value);
  }
  return out_archive;
}

template <typename T1, typename T2>
inline OutArchive& operator>>(OutArchive& out_archive,
                              ska::flat_hash_map<T1, T2>& m) {
  m.clear();
  size_t size;
  out_archive >> size;
  T1 key;
  T2 value;
  while (size--) {
    out_archive >> key >> value;
    m.emplace(key, value);
  }
  return out_archive;
}

}  // namespace grape

#endif  // GRAPE_SERIALIZATION_OUT_ARCHIVE_H_
