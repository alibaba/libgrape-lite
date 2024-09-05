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

#include <limits.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
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

inline bool exists_file(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

inline std::string get_absolute_path(const std::string& path) {
  char abs_path[PATH_MAX];
  if (realpath(path.c_str(), abs_path) == nullptr) {
    LOG(ERROR) << "Failed to get absolute path for " << path;
    return "";
  }
  return std::string(abs_path);
}

inline bool create_directories(const std::string& path) {
  char temp_path[256];
  snprintf(temp_path, sizeof(temp_path), "%s", path.c_str());

  for (char* p = temp_path + 1; *p; ++p) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir(temp_path, 0755) != 0 && errno != EEXIST) {
        std::cerr << "Error creating directory: " << temp_path << std::endl;
        return false;
      }
      *p = '/';
    }
  }
  if (mkdir(temp_path, 0755) != 0 && errno != EEXIST) {
    std::cerr << "Error creating directory: " << temp_path << std::endl;
    return false;
  }
  return true;
}

inline std::vector<std::string> split_string(const std::string& str,
                                             char delimiter) {
  std::vector<std::string> tokens;
  std::istringstream iss(str);
  std::string token;
  while (std::getline(iss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

inline size_t parse_size(const std::string& str) {
  std::istringstream iss(str);
  size_t digit;
  std::string tail;
  iss >> digit >> tail;
  if (tail.empty()) {
    return digit;
  } else if (tail == "kB") {
    return digit * 1024;
  } else {
    return 0;
  }
}

inline std::map<std::string, size_t> parse_meminfo() {
  std::ifstream mem_file("/proc/meminfo", std::ios_base::in);
  std::string line;
  std::map<std::string, size_t> ret;
  while (std::getline(mem_file, line)) {
    std::vector<std::string> parts = split_string(line, ':');
    assert(parts.size() == 2);
    ret[parts[0]] = parse_size(parts[1]);
  }
  return ret;
}

inline size_t get_available_memory() {
  auto meminfo = parse_meminfo();
#ifdef USE_HUGEPAGES
  return meminfo.at("HugePages_Free") * meminfo.at("Hugepagesize");
#else
  return meminfo.at("MemAvailable");
#endif
}

void show_thread_timing(const std::vector<double>& thread_time,
                        const std::string& prefix) {
  double total = 0, min_t = std::numeric_limits<double>::max(), max_t = 0;
  for (auto& t : thread_time) {
    total += t;
    min_t = std::min(min_t, t);
    max_t = std::max(max_t, t);
  }
  double avg_t = total / thread_time.size();
  LOG(INFO) << prefix << " min: " << min_t << " max: " << max_t
            << " avg: " << avg_t;
}

}  // namespace grape

#endif  // GRAPE_UTIL_H_
