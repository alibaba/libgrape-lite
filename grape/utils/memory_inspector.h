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

#ifndef GRAPE_UTILS_MEMORY_INSPECTOR_H_
#define GRAPE_UTILS_MEMORY_INSPECTOR_H_

#include "glog/logging.h"
#include "grape/utils/concurrent_queue.h"

namespace grape {

struct MemoryInspector {
  static MemoryInspector& GetInstance() {
    static MemoryInspector instance;
    return instance;
  }

  void allocate(size_t size) {
    lock_.lock();
    current_memory_usage += size;
    peak_memory_usage = std::max(current_memory_usage, peak_memory_usage);
    lock_.unlock();
  }

  void deallocate(size_t size) {
    lock_.lock();
    current_memory_usage -= size;
    lock_.unlock();
  }

  ~MemoryInspector() {
    double peak_memory_usage_gb = peak_memory_usage / 1024.0 / 1024.0 / 1024.0;
    LOG(INFO) << "Peak memory usage: " << peak_memory_usage_gb << " GB";
  }

  double GetCurrentMemoryUsage() const {
    return current_memory_usage / 1024.0 / 1024.0 / 1024.0;
  }
  double GetPeakMemoryUsage() const {
    return peak_memory_usage / 1024.0 / 1024.0 / 1024.0;
  }

 private:
  MemoryInspector() = default;

  size_t current_memory_usage = 0;
  size_t peak_memory_usage = 0;

  SpinLock lock_;
};

}  // namespace grape

#endif  // GRAPE_UTILS_MEMORY_INSPECTOR_H_
