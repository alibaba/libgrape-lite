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

#ifndef GRAPE_UTIL_H_
#define GRAPE_UTIL_H_

#ifndef __APPLE__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "grape/config.h"

namespace grape {

inline double GetCurrentTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

inline void GetMemoryUsage(const int proc_id, const std::string& info) {
  std::ifstream mem_file("/proc/self/stat", std::ios_base::in);
  std::string ignore;
  int64_t pagenum;
  uint64_t vm_byte;
  double vm_usage;
  const double BYTES_TO_MB = 1.0 / (1024 * 1024);
  mem_file >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
      ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
      ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
      ignore >> ignore >> vm_byte >> pagenum;
  vm_usage = vm_byte * BYTES_TO_MB;

  double pm_usage;
  pm_usage = pagenum * getpagesize() * BYTES_TO_MB;
  VLOG(2) << "[pid=" << proc_id << "] alloc_size: " << pm_usage << "MB, "
          << vm_usage << "MB. " << info;
}

template <typename... Args>
inline std::string StringFormat(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

/**
 * @brief Get the formatted result filename.
 *
 */
inline std::string GetResultFilename(const std::string& prefix,
                                     const fid_t fid) {
  return StringFormat("%s/result_frag_%s", prefix.c_str(),
                      std::to_string(fid).c_str());
}

/**
 * @brief sort the target vector and eliminate the duplicated elements.
 *
 * @tparam T
 * @param vec to be sorted.
 */
template <typename T>
void DistinctSort(std::vector<T>& vec) {
  std::sort(vec.begin(), vec.end());
  size_t size = vec.size();
  size_t count = 0;
  for (size_t i = 1; i < size; ++i) {
    if (vec[i] == vec[i - 1]) {
      ++count;
    } else {
      vec[i - count] = vec[i];
    }
  }
  vec.resize(size - count);
}

template <typename T>
struct IdHasher {};

template <>
struct IdHasher<uint32_t> {
  static uint32_t hash(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
  }
};

template <>
struct IdHasher<uint64_t> {
  static uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
  }
};

template <typename T>
struct IdenticalHasher {};

template <>
struct IdenticalHasher<uint32_t> {
  static uint32_t hash(uint32_t x) { return x; }
};

template <>
struct IdenticalHasher<uint64_t> {
  static uint64_t hash(uint64_t x) { return x; }
};

static inline bool exists_file(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

}  // namespace grape

#endif  // GRAPE_UTIL_H_
