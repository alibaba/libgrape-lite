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

#ifndef GRAPE_UTILS_BITSET_H_
#define GRAPE_UTILS_BITSET_H_

#include <stdlib.h>

#include <algorithm>
#include <utility>

#define WORD_SIZE(n) (((n) + 63ul) >> 6)

#define WORD_INDEX(i) ((i) >> 6)
#define BIT_OFFSET(i) ((i) &0x3f)

#define ROUND_UP(i) (((i) + 63ul) & (~63ul))
#define ROUND_DOWN(i) ((i) & (~63ul))

namespace grape {

/**
 * @brief Bitset is a highly-optimized bitset implementation.
 */
class Bitset {
 public:
  Bitset() : data_(NULL), size_(0), size_in_words_(0) {}
  explicit Bitset(size_t size) : size_(size) {
    size_in_words_ = WORD_SIZE(size_);
    data_ = static_cast<uint64_t*>(malloc(size_in_words_ * sizeof(uint64_t)));
    clear();
  }
  ~Bitset() {
    if (data_ != NULL) {
      free(data_);
    }
  }

  void init(size_t size) {
    if (data_ != NULL) {
      free(data_);
    }
    size_ = size;
    size_in_words_ = WORD_SIZE(size_);
    data_ = static_cast<uint64_t*>(malloc(size_in_words_ * sizeof(uint64_t)));
    clear();
  }

  void clear() {
    for (size_t i = 0; i < size_in_words_; ++i) {
      data_[i] = 0;
    }
  }

  void parallel_clear(int thread_num) {
#pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i < size_in_words_; ++i) {
      data_[i] = 0;
    }
  }

  bool empty() const {
    for (size_t i = 0; i < size_in_words_; ++i) {
      if (data_[i]) {
        return false;
      }
    }
    return true;
  }

  bool partial_empty(size_t begin, size_t end) const {
    end = std::min(end, size_);
    size_t cont_beg = ROUND_UP(begin);
    size_t cont_end = ROUND_DOWN(end);
    size_t word_beg = WORD_INDEX(cont_beg);
    size_t word_end = WORD_INDEX(cont_end);
    for (size_t i = word_beg; i < word_end; ++i) {
      if (data_[i]) {
        return false;
      }
    }
    if (cont_beg != begin) {
      uint64_t first_word = data_[WORD_INDEX(begin)];
      first_word = (first_word >> (64 - (cont_beg - begin)));
      if (first_word) {
        return false;
      }
    }
    if (cont_end != end) {
      uint64_t last_word = data_[WORD_INDEX(end)];
      last_word = (last_word & ((1ul << (end - cont_end)) - 1));
      if (last_word) {
        return false;
      }
    }
    return true;
  }

  bool get_bit(size_t i) const {
    return data_[WORD_INDEX(i)] & (1ul << BIT_OFFSET(i));
  }

  void set_bit(size_t i) {
    __sync_fetch_and_or(data_ + WORD_INDEX(i), 1ul << BIT_OFFSET(i));
  }

  bool set_bit_with_ret(size_t i) {
    uint64_t mask = 1ul << BIT_OFFSET(i);
    uint64_t ret = __sync_fetch_and_or(data_ + WORD_INDEX(i), mask);
    return !(ret & mask);
  }

  void reset_bit(size_t i) {
    __sync_fetch_and_and(data_ + WORD_INDEX(i), ~(1ul << BIT_OFFSET(i)));
  }

  bool reset_bit_with_ret(size_t i) {
    uint64_t mask = 1ul << BIT_OFFSET(i);
    uint64_t ret = __sync_fetch_and_and(data_ + WORD_INDEX(i), ~mask);
    return (ret & mask);
  }

  void swap(Bitset& other) {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
    std::swap(size_in_words_, other.size_in_words_);
  }

  size_t count() const {
    size_t ret = 0;
    for (size_t i = 0; i < size_in_words_; ++i) {
      ret += __builtin_popcountll(data_[i]);
    }
    return ret;
  }

  size_t parallel_count(int thread_num) const {
    size_t ret = 0;
#pragma omp parallel for num_threads(thread_num) reduction(+ : ret)
    for (size_t i = 0; i < size_in_words_; ++i) {
      ret += __builtin_popcountll(data_[i]);
    }
    return ret;
  }

  size_t partial_count(size_t begin, size_t end) const {
    size_t ret = 0;
    size_t cont_beg = ROUND_UP(begin);
    size_t cont_end = ROUND_DOWN(end);
    size_t word_beg = WORD_INDEX(cont_beg);
    size_t word_end = WORD_INDEX(cont_end);
    for (size_t i = word_beg; i < word_end; ++i) {
      ret += __builtin_popcountll(data_[i]);
    }
    if (cont_beg != begin) {
      uint64_t first_word = data_[WORD_INDEX(begin)];
      first_word = (first_word >> (64 - (cont_beg - begin)));
      ret += __builtin_popcountll(first_word);
    }
    if (cont_end != end) {
      uint64_t last_word = data_[WORD_INDEX(end)];
      last_word = (last_word & ((1ul << (end - cont_end)) - 1));
      ret += __builtin_popcountll(last_word);
    }
    return ret;
  }

  size_t parallel_partial_count(int thread_num, size_t begin,
                                size_t end) const {
    size_t ret = 0;
    size_t cont_beg = ROUND_UP(begin);
    size_t cont_end = ROUND_DOWN(end);
    size_t word_beg = WORD_INDEX(cont_beg);
    size_t word_end = WORD_INDEX(cont_end);
#pragma omp parallel for num_threads(thread_num) reduction(+ : ret)
    for (size_t i = word_beg; i < word_end; ++i) {
      ret += __builtin_popcountll(data_[i]);
    }
    if (cont_beg != begin) {
      uint64_t first_word = data_[WORD_INDEX(begin)];
      first_word = (first_word >> (64 - (cont_beg - begin)));
      ret += __builtin_popcountll(first_word);
    }
    if (cont_end != end) {
      uint64_t last_word = data_[WORD_INDEX(end)];
      last_word = (last_word & ((1ul << (end - cont_end)) - 1));
      ret += __builtin_popcountll(last_word);
    }
    return ret;
  }

  inline uint64_t get_word(size_t i) const { return data_[WORD_INDEX(i)]; }

  inline const uint64_t* get_word_ptr(size_t i) const {
    return &data_[WORD_INDEX(i)];
  }

 private:
  uint64_t* data_;
  size_t size_;
  size_t size_in_words_;
};

class RefBitset {
 public:
  RefBitset() : data(NULL) {}
  RefBitset(void* d, size_t b, size_t e) {
    data = static_cast<uint64_t*>(d);
    begin = b / 64 * 64;
    end = (e + 63) / 64 * 64;

    data[0] &= (~((1ul << (b - begin)) - 1ul));
    data[((end - begin) / 64) - 1] &= ((1ul << (64 - (end - e))) - 1ul);
  }
  ~RefBitset() {}

  bool get_bit(size_t loc) const {
    return data[WORD_INDEX(loc - begin)] & (1ul << BIT_OFFSET(loc));
  }

  uint64_t get_word_by_index(size_t index) { return data[index]; }

  size_t get_word_num() const { return (end - begin) / 64; }

  uint64_t* data;
  size_t begin;
  size_t end;
};

#undef WORD_SIZE
#undef WORD_INDEX
#undef BIT_OFFSET
#undef ROUND_UP
#undef ROUND_DOWN

}  // namespace grape

#endif  // GRAPE_UTILS_BITSET_H_
