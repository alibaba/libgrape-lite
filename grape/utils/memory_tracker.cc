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

#include "grape/utils/memory_tracker.h"

namespace grape {

#ifdef TRACKING_MEMORY

MemoryTracker& MemoryTracker::GetInstance() {
  static MemoryTracker instance;
  return instance;
}

MemoryTracker::MemoryTracker()
    : current_memory_usage_(0), peak_memory_usage_(0) {}
MemoryTracker::~MemoryTracker() {}

void MemoryTracker::allocate(size_t size) {
  std::lock_guard<SpinLock> lock(lock_);
  current_memory_usage_ += size;
  peak_memory_usage_ = std::max(current_memory_usage_, peak_memory_usage_);
}

void MemoryTracker::deallocate(size_t size) {
  std::lock_guard<SpinLock> lock(lock_);
  current_memory_usage_ =
      (current_memory_usage_ > size) ? (current_memory_usage_ - size) : 0;
}

size_t MemoryTracker::GetCurrentMemoryUsage() const {
  return current_memory_usage_;
}

size_t MemoryTracker::GetPeakMemoryUsage() const { return peak_memory_usage_; }

std::string MemoryTracker::GetMemoryUsageInfo() const {
  std::lock_guard<SpinLock> lock(lock_);
  return "Current memory usage: " +
         std::to_string(current_memory_usage_ / 1024.0 / 1024.0 / 1024.0) +
         " GB, peak memory usage: " +
         std::to_string(peak_memory_usage_ / 1024.0 / 1024.0 / 1024.0) + " GB.";
}

#endif

}  // namespace grape
