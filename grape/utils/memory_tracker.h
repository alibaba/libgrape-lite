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

#ifndef GRAPE_UTILS_MEMORY_TRACKER_H_
#define GRAPE_UTILS_MEMORY_TRACKER_H_

#include "grape/utils/concurrent_queue.h"

namespace grape {

#define TRACKING_MEMORY_ALLOCATIONS
#ifdef TRACKING_MEMORY_ALLOCATIONS

struct MemoryTracker {
  static MemoryTracker& GetInstance();

  void allocate(size_t size);
  void deallocate(size_t size);

  size_t GetCurrentMemoryUsage() const;
  size_t GetPeakMemoryUsage() const;
  std::string GetMemoryUsageInfo() const;

 private:
  MemoryTracker();
  ~MemoryTracker();

  size_t current_memory_usage_;
  size_t peak_memory_usage_;
  mutable SpinLock lock_;
};

#endif

}  // namespace grape

#endif  // GRAPE_UTILS_MEMORY_TRACKER_H_
