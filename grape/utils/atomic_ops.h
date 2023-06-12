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
  __atomic_fetch_add(&a, b, __ATOMIC_RELAXED);
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

/**
 * @brief Atomic sub a value. Equavalent to:
 *
 * \code
 * a -= b;
 * \endcode
 *
 * @tparam T Type of the operands.
 * @param a Object to process.
 * @param b Value to sub.
 */
template <typename T>
inline void atomic_sub(T& a, T b) {
  __atomic_fetch_sub(&a, b, __ATOMIC_RELAXED);
}

template <>
inline void atomic_sub(float& a, float b) {
  volatile float new_a, old_a;
  do {
    old_a = a;
    new_a = old_a - b;
  } while (!atomic_compare_and_swap(a, old_a, new_a));
}

template <>
inline void atomic_sub(double& a, double b) {
  volatile double new_a, old_a;
  do {
    old_a = a;
    new_a = old_a - b;
  } while (!atomic_compare_and_swap(a, old_a, new_a));
}

}  // namespace grape

#endif  // GRAPE_UTILS_ATOMIC_OPS_H_
