/** Copyright 2021 Alibaba Group Holding Limited.

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

#ifndef GRAPE_PARALLEL_PARALLEL_ENGINE_SPEC_H_
#define GRAPE_PARALLEL_PARALLEL_ENGINE_SPEC_H_

#include "grape/worker/comm_spec.h"

namespace grape {

struct ParallelEngineSpec {
  uint32_t thread_num;
  bool affinity;
  std::vector<uint32_t> cpu_list;
};

inline ParallelEngineSpec DefaultParallelEngineSpec() {
  ParallelEngineSpec spec;
  spec.thread_num = std::thread::hardware_concurrency();
  spec.affinity = false;
  spec.cpu_list.clear();
  return spec;
}

inline ParallelEngineSpec MultiProcessSpec(const CommSpec& comm_spec,
                                           bool affinity = false) {
  ParallelEngineSpec spec;
  uint32_t total_thread_num = std::thread::hardware_concurrency();
  uint32_t each_process_thread_num =
      (total_thread_num + comm_spec.local_num() - 1) / comm_spec.local_num();
  spec.thread_num = each_process_thread_num;
  spec.affinity = affinity;
  spec.cpu_list.clear();
  if (affinity) {
    uint32_t offset = each_process_thread_num * comm_spec.local_id();
    for (uint32_t i = 0, j = 0; i < each_process_thread_num; ++i, ++j) {
      if (offset + j == total_thread_num)
        j = 0;
      spec.cpu_list.push_back(offset + j);
    }
  }
  return spec;
}

}  // namespace grape

#endif  // GRAPE_PARALLEL_PARALLEL_ENGINE_SPEC_H_
