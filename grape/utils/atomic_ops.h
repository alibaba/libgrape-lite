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

/**
 * @file atomic_ops.h
 *
 * Some atomic instructions.
 */

#ifndef GRAPE_UTILS_ATOMIC_OPS_H_
#define GRAPE_UTILS_ATOMIC_OPS_H_

#include <stdint.h>

namespace grape {

/**
 * @brief Atomic compare and swap operation. Equavalent to:
 *
 * \code
 * if (val == old_val) {
 *   val = new_val;
 *   return true;
 * } else {
 *   return false;
 * }
 * \endcode
 *
 * @tparam T Type of the operands.
 * @param val Object to process.
 * @param old_val Old value to check.
 * @param new_val New value to assign.
 *
 * @return Whether the value has been changed.
 */
template <typename T>
inline bool atomic_compare_and_swap(T& val, T old_val, T new_val) {
  return __sync_bool_compare_and_swap(&val, old_val, new_val);
}

template <>
inline bool atomic_compare_and_swap(float& val, float old_val, float new_val) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(&val);
  const uint32_t* old_val_ptr = reinterpret_cast<const uint32_t*>(&old_val);
  const uint32_t* new_val_ptr = reinterpret_cast<const uint32_t*>(&new_val);
  return __sync_bool_compare_and_swap(ptr, *old_val_ptr, *new_val_ptr);
}

template <>
inline bool atomic_compare_and_swap(double& val, double old_val,
                                    double new_val) {
  uint64_t* ptr = reinterpret_cast<uint64_t*>(&val);
  const uint64_t* old_val_ptr = reinterpret_cast<const uint64_t*>(&old_val);
  const uint64_t* new_val_ptr = reinterpret_cast<const uint64_t*>(&new_val);
  return __sync_bool_compare_and_swap(ptr, *old_val_ptr, *new_val_ptr);
}

/**
 * @brief Atomic compare and store the minimum value. Equavalent to:
 *
 * \code
 * if (a > b) {
 *   a = b;
 *   return true;
 * } else {
 *   return false;
 * }
 * \endcode
 *
 * @tparam T Type of the operands.
 * @param a Object to process.
 * @param b Value to compare.
 *
 * @return Whether the value has been changed.
 */
template <typename T>
inline bool atomic_min(T& a, T b) {
  volatile T curr_a;
  bool done = false;
  do {
    curr_a = a;
  } while (curr_a > b && !(done = atomic_compare_and_swap(a, curr_a, b)));
  return done;
}
/**
 * @brief Atomic compare and store the minimum value. Equavalent to:
 *
 * \code
 * if (a < b) {
 *   a = b;
 *   return true;
 * } else {
 *   return false;
 * }
 * \endcode
 *
 * @tparam T Type of the operands.
 * @param a Object to process.
 * @param b Value to compare.
 *
 * @return Whether the value has been changed.
 */
template <typename T>
inline bool atomic_max(T& a, T b) {
  volatile T curr_a;
  bool done = false;
  do {
    curr_a = a;
  } while (curr_a < b && !(done = atomic_compare_and_swap(a, curr_a, b)));
  return done;
}
template <class ET>
inline bool CAS(ET* ptr, ET oldv, ET newv) {
  if (sizeof(ET) == 1) {
    return __sync_bool_compare_and_swap(reinterpret_cast<char*>(ptr),
                                        *(reinterpret_cast<char*>(&oldv)),
                                        *(reinterpret_cast<char*>(&newv)));
  } else if (sizeof(ET) == 4) {
    return __sync_bool_compare_and_swap(reinterpret_cast<int32_t*>(ptr),
                                        *(reinterpret_cast<int32_t*>(&oldv)),
                                        *(reinterpret_cast<int32_t*>(&newv)));
  } else if (sizeof(ET) == 8) {
    return __sync_bool_compare_and_swap(reinterpret_cast<int64_t*>(ptr),
                                        *(reinterpret_cast<int64_t*>(&oldv)),
                                        *(reinterpret_cast<int64_t*>(&newv)));
  } else if (sizeof(ET) == 16) {
    if (sizeof(long double) != 16) {
      std::cout << "Unsupported platform" << std::endl;
      std::abort();
    }
    return __atomic_compare_exchange(reinterpret_cast<long double*>(ptr),
                                     reinterpret_cast<long double*>(&oldv),
                                     reinterpret_cast<long double*>(&newv),
                                     true, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
  } else {
    std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
    std::abort();
  }
}

/**
 * @brief Atomic add a value. Equavalent to:
 *
 * \code
 * a += b;
 * \endcode
 *
 * @tparam T Type of the operands.
 * @param a Object to process.
 * @param b Value to add.
 */
template <typename T>
inline void atomic_add(T& a, T b) {
  __sync_fetch_and_add(&a, b);
}

template <>
inline void atomic_add(float& a, float b) {
  volatile float new_a, old_a;
  do {
    old_a = a;
    new_a = old_a + b;
  } while (!atomic_compare_and_swap(a, old_a, new_a));
}

template <>
inline void atomic_add(double& a, double b) {
  volatile double new_a, old_a;
  do {
    old_a = a;
    new_a = old_a + b;
  } while (!atomic_compare_and_swap(a, old_a, new_a));
}

inline float atomic_exch(float& a, float b) {
  auto* ptr = reinterpret_cast<uint32_t*>(&a);
  auto* new_val_ptr = reinterpret_cast<uint32_t*>(&b);
  uint32_t ret;
  __atomic_exchange(ptr, new_val_ptr, &ret, __ATOMIC_SEQ_CST);
  return *reinterpret_cast<float*>(&ret);
}

inline double atomic_exch(double& a, double b) {
  auto* ptr = reinterpret_cast<double*>(&a);
  auto* new_val_ptr = reinterpret_cast<double*>(&b);
  double ret;
  __atomic_exchange(ptr, new_val_ptr, &ret, __ATOMIC_SEQ_CST);
  return ret;
}

inline int32_t atomic_exch(int32_t& a, int32_t b) {
  auto* ptr = reinterpret_cast<int32_t*>(&a);
  auto* new_val_ptr = reinterpret_cast<int32_t*>(&b);
  int32_t ret;
  __atomic_exchange(ptr, new_val_ptr, &ret, __ATOMIC_SEQ_CST);
  return ret;
}

inline uint32_t atomic_exch(uint32_t& a, uint32_t b) {
  auto* ptr = reinterpret_cast<uint32_t*>(&a);
  auto* new_val_ptr = reinterpret_cast<uint32_t*>(&b);
  uint32_t ret;
  __atomic_exchange(ptr, new_val_ptr, &ret, __ATOMIC_SEQ_CST);
  return ret;
}

}  // namespace grape

#endif  // GRAPE_UTILS_ATOMIC_OPS_H_
