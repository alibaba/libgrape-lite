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

#ifndef GRAPE_SERIALIZATION_FIXED_IN_ARCHIVE_H_
#define GRAPE_SERIALIZATION_FIXED_IN_ARCHIVE_H_

#include "grape/serialization/in_archive.h"
#include "grape/utils/message_buffer_pool.h"

namespace grape {

template <typename T>
struct SerializedSize {
  static size_t size(const T& v) {
    LOG(FATAL) << "Not implemented";
    return sizeof(T);
  }
};

template <>
struct SerializedSize<EmptyType> {
  static size_t size(const EmptyType& v) {
    LOG(FATAL) << "Not expected to be called";
    return 0;
  }
};

template <>
struct SerializedSize<std::string> {
  static size_t size(const std::string& v) { return v.size() + sizeof(size_t); }
};

template <>
struct SerializedSize<nonstd::string_view> {
  static size_t size(const nonstd::string_view& v) {
    return v.size() + sizeof(size_t);
  }
};

class FixedInArchive : public Allocator<char> {
 public:
  FixedInArchive() : begin_(0), offset_(0) {}
  FixedInArchive(FixedInArchive&& rhs)
      : buffer_(std::move(rhs.buffer_)),
        begin_(rhs.begin_),
        offset_(rhs.offset_) {
    rhs.begin_ = 0;
    rhs.offset_ = 0;
  }
  ~FixedInArchive() {}

  void init(MessageBuffer&& buf) {
    buffer_ = std::move(buf);
    begin_ = 0;
    offset_ = 0;
  }

  char* data() { return buffer_.buffer; }
  const char* data() const { return buffer_.buffer; }

  size_t size() const { return offset_ - begin_; }
  size_t used() const { return offset_; }
  size_t remaining() const { return buffer_.size - offset_; }

  void add_byte(char v) {
    buffer_.buffer[offset_] = v;
    offset_++;
  }

  void add_bytes(const void* bytes, size_t n) {
    memcpy(buffer_.buffer + offset_, bytes, n);
    offset_ += n;
  }

  void swap(FixedInArchive& rhs) {
    std::swap(buffer_, rhs.buffer_);
    std::swap(begin_, rhs.begin_);
    std::swap(offset_, rhs.offset_);
  }

  MicroBuffer take() {
    MicroBuffer ret(buffer_.buffer + begin_, offset_ - begin_);
    begin_ = offset_;
    return ret;
  }

  void reset() { begin_ = offset_ = 0; }

  MessageBuffer& buffer() { return buffer_; }
  const MessageBuffer& buffer() const { return buffer_; }

 private:
  MessageBuffer buffer_;
  size_t begin_;
  size_t offset_;
};

template <typename T,
          typename std::enable_if<std::is_pod<T>::value, T>::type* = nullptr>
inline FixedInArchive& operator<<(FixedInArchive& arc, T u) {
  arc.add_bytes(&u, sizeof(T));
  return arc;
}

inline FixedInArchive& operator<<(FixedInArchive& arc, const EmptyType& v) {
  return arc;
}

}  // namespace grape

#endif  // GRAPE_SERIALIZATION_FIXED_IN_ARCHIVE_H_
