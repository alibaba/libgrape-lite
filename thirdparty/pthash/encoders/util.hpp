/** Copyright 2020-2024 Giulio Ermanno Pibiri and Roberto Trani
 *
 * The following sets forth attribution notices for third party software.
 *
 * PTHash:
 * The software includes components licensed by Giulio Ermanno Pibiri and
 * Roberto Trani, available at https://github.com/jermp/pthash
 *
 * Licensed under the MIT License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cassert>
#include <cstdint>

#if defined(__x86_64__) && __SSE4_2__
#include <immintrin.h>
#endif

namespace pthash::util {

#if defined(__x86_64__) && __SSE4_2__
template <typename T>
inline void prefetch(T const* ptr) {
  _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
}
#endif

inline uint8_t msb(uint64_t x) {
  assert(x);
  unsigned long ret = -1U;
  if (x) {
    ret = (unsigned long) (63 - __builtin_clzll(x));
  }
  return (uint8_t) ret;
}

inline bool bsr64(unsigned long* const index, const uint64_t mask) {
  if (mask) {
    *index = (unsigned long) (63 - __builtin_clzll(mask));
    return true;
  } else {
    return false;
  }
}

inline uint8_t msb(uint64_t x, unsigned long& ret) { return bsr64(&ret, x); }

inline uint8_t lsb(uint64_t x, unsigned long& ret) {
  if (x) {
    ret = (unsigned long) __builtin_ctzll(x);
    return true;
  }
  return false;
}

inline uint8_t lsb(uint64_t x) {
  assert(x);
  unsigned long ret = -1U;
  lsb(x, ret);
  return (uint8_t) ret;
}

inline uint64_t popcount(uint64_t x) {
#ifdef __SSE4_2__
  return static_cast<uint64_t>(_mm_popcnt_u64(x));
#elif __cplusplus >= 202002L
  return std::popcount(x);
#else
  return static_cast<uint64_t>(__builtin_popcountll(x));
#endif
}

inline uint64_t select64_pdep_tzcnt(uint64_t x, const uint64_t k) {
#if defined(__x86_64__) && defined(__BMI2__) || defined(__AVX2__)
  uint64_t i = 1ULL << k;
  asm("pdep %[x], %[mask], %[x]" : [x] "+r"(x) : [mask] "r"(i));
  asm("tzcnt %[bit], %[index]" : [index] "=r"(i) : [bit] "g"(x) : "cc");
  return i;
#else
  uint64_t count = 0;
  uint64_t result = 0;

  for (uint64_t bit = 0; bit < 64; ++bit) {
    if ((x >> bit) & 1) {
      if (count == k) {
        result = bit;
        break;
      }
      ++count;
    }
  }

  return result;
#endif
}

inline uint64_t select_in_word(const uint64_t x, const uint64_t k) {
  assert(k < popcount(x));
  return select64_pdep_tzcnt(x, k);
}

}  // namespace pthash::util