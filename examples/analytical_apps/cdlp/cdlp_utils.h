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

#ifndef EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_UTILS_H_
#define EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_UTILS_H_

#include <grape/grape.h>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

namespace grape {
template <typename LABEL_T, typename VERTEX_ARRAY_T, typename ADJ_LIST_T>
inline LABEL_T update_label_fast(const ADJ_LIST_T& edges,
                                 const VERTEX_ARRAY_T& labels) {
  std::vector<LABEL_T> local_labels;
  for (auto& e : edges) {
    local_labels.emplace_back(labels[e.get_neighbor()]);
  }
  std::sort(local_labels.begin(), local_labels.end());

  LABEL_T curr_label = local_labels[0];
  int curr_count = 1;
  LABEL_T best_label = LABEL_T{};
  int best_count = 0;
  int label_num = local_labels.size();

  for (int i = 1; i < label_num; ++i) {
    if (local_labels[i] != local_labels[i - 1]) {
      if (curr_count > best_count) {
        best_label = curr_label;
        best_count = curr_count;
      }
      curr_label = local_labels[i];
      curr_count = 1;
    } else {
      ++curr_count;
    }
  }

  if (curr_count > best_count) {
    return curr_label;
  } else {
    return best_label;
  }
}

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_UTILS_H_
