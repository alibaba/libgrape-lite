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

#ifndef GRAPE_UTILS_VARINT_H_
#define GRAPE_UTILS_VARINT_H_

#include <stdint.h>

#include <vector>

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape {

inline int varint_length(uint64_t value) {
  static constexpr uint64_t B = 128;
  int len = 1;
  while (value >= B) {
    len++;
    value >>= 7;
  }
  return len;
}

class VarintEncoder {
 public:
  VarintEncoder() = default;
  ~VarintEncoder() = default;

  void encode_u32(uint32_t v) {
    static constexpr uint32_t B = 128;
    if (v < (1 << 7)) {
      buf_.push_back(v);
    } else if (v < (1 << 14)) {
      buf_.push_back(v | B);
      buf_.push_back(v >> 7);
    } else if (v < (1 << 21)) {
      buf_.push_back(v | B);
      buf_.push_back((v >> 7) | B);
      buf_.push_back(v >> 14);
    } else if (v < (1 << 28)) {
      buf_.push_back(v | B);
      buf_.push_back((v >> 7) | B);
      buf_.push_back((v >> 14) | B);
      buf_.push_back(v >> 21);
    } else {
      buf_.push_back(v | B);
      buf_.push_back((v >> 7) | B);
      buf_.push_back((v >> 14) | B);
      buf_.push_back((v >> 21) | B);
      buf_.push_back(v >> 28);
    }
  }
  void encode_u64(uint64_t v) {
    static constexpr uint64_t B = 128;
    while (v >= B) {
      buf_.push_back(v | B);
      v >>= 7;
    }
    buf_.push_back(v);
  }
  void reserve(size_t size) { buf_.reserve(size); }
  void clear() { buf_.clear(); }
  size_t size() const { return buf_.size(); }

  const uint8_t* data() const { return buf_.data(); }

  bool empty() const { return buf_.empty(); }

 private:
  std::vector<uint8_t> buf_;
};

class VarintDecoder {
 public:
  VarintDecoder(const char* p, size_t size)
      : p_(reinterpret_cast<const uint8_t*>(p)), limit_(p_ + size) {}
  ~VarintDecoder() = default;

  void reset(const char* p, size_t size) {
    p_ = reinterpret_cast<const uint8_t*>(p);
    limit_ = p_ + size;
  }

  bool empty() const { return p_ == limit_; }

  uint32_t decode_u32() {
    uint32_t result = 0;
    for (uint32_t shift = 0; shift <= 28 && p_ < limit_; shift += 7) {
      uint32_t b = *p_;
      p_++;
      if (b & 128) {
        result |= (b & 127) << shift;
      } else {
        result |= b << shift;
        return result;
      }
    }
    return result;
  }

  uint64_t decode_u64() {
    uint64_t result = 0;
    for (uint32_t shift = 0; shift <= 63 && p_ < limit_; shift += 7) {
      uint64_t b = *p_;
      p_++;
      if (b & 128) {
        result |= (b & 127) << shift;
      } else {
        result |= b << shift;
        return result;
      }
    }
    return result;
  }

  size_t size() const { return limit_ - p_; }

  const uint8_t* data() const { return p_; }

  const uint8_t* limit() const { return limit_; }

 private:
  const uint8_t* p_;
  const uint8_t* limit_;
};

template <typename T>
struct VarintUtil {
  static void encode(T v, VarintEncoder& encoder) {}

  static T decode(VarintDecoder& decoder) { return T(); }

  static const uint8_t* decode_raw(const uint8_t* p, T& val) { return p; }
};

template <>
struct VarintUtil<uint32_t> {
  static void encode(uint32_t v, VarintEncoder& encoder) {
    encoder.encode_u32(v);
  }
  static bool decode(VarintDecoder& decoder, uint32_t& val) {
    if (decoder.empty()) {
      return false;
    }
    val = decoder.decode_u32();
    return true;
  }

  static const uint8_t* decode_raw(const uint8_t* p, uint32_t& val) {
    val = 0;
    for (uint32_t shift = 0; shift <= 28; shift += 7) {
      uint32_t b = *p;
      p++;
      if (b & 128) {
        val |= (b & 127) << shift;
      } else {
        val |= b << shift;
        return p;
      }
    }
    return p;
  }

  static void encode_to_archive(InArchive& arc, uint32_t v) {
    static constexpr uint32_t B = 128;
    if (v < (1 << 7)) {
      arc.AddByte(v);
    } else if (v < (1 << 14)) {
      arc.AddByte(v | B);
      arc.AddByte(v >> 7);
    } else if (v < (1 << 21)) {
      arc.AddByte(v | B);
      arc.AddByte((v >> 7) | B);
      arc.AddByte(v >> 14);
    } else if (v < (1 << 28)) {
      arc.AddByte(v | B);
      arc.AddByte((v >> 7) | B);
      arc.AddByte((v >> 14) | B);
      arc.AddByte(v >> 21);
    } else {
      arc.AddByte(v | B);
      arc.AddByte((v >> 7) | B);
      arc.AddByte((v >> 14) | B);
      arc.AddByte((v >> 21) | B);
      arc.AddByte(v >> 28);
    }
  }

  static void decode_from_archive(OutArchive& arc, uint32_t& val) {
    val = 0;
    uint8_t byte;
    for (uint32_t shift = 0; shift <= 28; shift += 7) {
      arc >> byte;
      uint32_t b = byte;
      if (b & 128) {
        val |= (b & 127) << shift;
      } else {
        val |= b << shift;
        break;
      }
    }
  }
};

template <>
struct VarintUtil<uint64_t> {
  static void encode(uint64_t v, VarintEncoder& encoder) {
    encoder.encode_u64(v);
  }
  static bool decode(VarintDecoder& decoder, uint64_t& val) {
    if (decoder.empty()) {
      return false;
    }
    val = decoder.decode_u64();
    return true;
  }

  static const uint8_t* decode_raw(const uint8_t* p, uint64_t& val) {
    val = 0;
    for (uint32_t shift = 0; shift <= 63; shift += 7) {
      uint64_t b = *p;
      p++;
      if (b & 128) {
        val |= (b & 127) << shift;
      } else {
        val |= b << shift;
        return p;
      }
    }
    return p;
  }

  static void encode_to_archive(InArchive& arc, uint64_t v) {
    static constexpr uint64_t B = 128;
    while (v >= B) {
      arc.AddByte(static_cast<char>(v | B));
      v >>= 7;
    }
    arc.AddByte(static_cast<char>(v));
  }

  static void decode_from_archive(OutArchive& arc, uint64_t& val) {
    val = 0;
    uint8_t byte;
    for (uint32_t shift = 0; shift <= 63; shift += 7) {
      arc >> byte;
      uint64_t b = byte;
      if (b & 128) {
        val |= (b & 127) << shift;
      } else {
        val |= b << shift;
        break;
      }
    }
  }
};

template <typename T>
class DeltaVarintEncoder {
 public:
  DeltaVarintEncoder() = default;
  ~DeltaVarintEncoder() = default;

  void push_back(T v) {
    VarintUtil<T>::encode(v - last_, encoder_);
    last_ = v;
  }

  void reserve(size_t size) { encoder_.reserve(size * sizeof(T)); }

  bool empty() const { return encoder_.empty(); }

  void clear() {
    encoder_.clear();
    last_ = 0;
  }

  size_t size() const { return encoder_.size(); }

  const char* data() const {
    return reinterpret_cast<const char*>(encoder_.data());
  }

  const VarintEncoder& encoder() const { return encoder_; }

 private:
  VarintEncoder encoder_;
  T last_ = 0;
};

template <typename T>
class DeltaVarintDecoder {
 public:
  DeltaVarintDecoder() : decoder_(nullptr, 0), last_(0) {}
  DeltaVarintDecoder(const char* p, size_t size)
      : decoder_(p, size), last_(0) {}
  ~DeltaVarintDecoder() = default;

  void reset(const char* p, size_t size) {
    decoder_.reset(p, size);
    last_ = 0;
  }

  bool pop(T& v) {
    T delta;
    if (VarintUtil<T>::decode(decoder_, delta)) {
      last_ += delta;
      v = last_;
      return true;
    }
    return false;
  }

  size_t size() const { return decoder_.size(); }

  VarintDecoder& decoder() { return decoder_; }

  void reset_last() { last_ = 0; }

 private:
  VarintDecoder decoder_;
  T last_;
};

InArchive& operator<<(InArchive& arc, const VarintEncoder& encoder) {
  VarintUtil<uint64_t>::encode_to_archive(arc, encoder.size());
  arc.AddBytes(encoder.data(), encoder.size());
  return arc;
}

OutArchive& operator>>(OutArchive& arc, VarintDecoder& decoder) {
  uint64_t size;
  VarintUtil<uint64_t>::decode_from_archive(arc, size);
  decoder.reset(static_cast<const char*>(arc.GetBytes(size)), size);
  return arc;
}

template <typename T>
InArchive& operator<<(InArchive& arc, const DeltaVarintEncoder<T>& encoder) {
  arc << encoder.encoder();
  return arc;
}

template <typename T>
OutArchive& operator>>(OutArchive& arc, DeltaVarintDecoder<T>& decoder) {
  arc >> decoder.decoder();
  decoder.reset_last();
  return arc;
}

}  // namespace grape

#endif  // GRAPE_UTILS_VARINT_H_
