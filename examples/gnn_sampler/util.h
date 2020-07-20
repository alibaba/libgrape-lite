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

#ifndef EXAMPLES_GNN_SAMPLER_UTIL_H_
#define EXAMPLES_GNN_SAMPLER_UTIL_H_

#include <grape/grape.h>
#include <grape/io/line_parser_base.h>

enum class RandomStrategy { Random, EdgeWeight, TopK };

void split(const std::string& str, char delim,
           std::vector<std::string>& ret_strs) {
  ret_strs.clear();
  size_t start;
  size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    ret_strs.push_back(str.substr(start, end - start));
  }
}

void parse_hop_and_num(const std::string& hop_and_num_str,
                       std::vector<uint32_t>& nums_of_hop,
                       std::vector<uint32_t>& hop_size) {
  std::vector<std::string> hop_params;
  split(hop_and_num_str, '-', hop_params);
  for (auto& hop : hop_params) {
    nums_of_hop.push_back(std::stoul(hop));
  }
  hop_size.resize(nums_of_hop.size() + 1);
  hop_size[0] = 1;
  for (size_t i = 0; i < nums_of_hop.size(); ++i) {
    hop_size[i + 1] = hop_size[i] * nums_of_hop[i];
  }
  hop_size[0] = 0;
  for (size_t i = 1; i < hop_size.size(); ++i) {
    hop_size[i] = hop_size[i - 1] + hop_size[i];
  }
}

#endif  // EXAMPLES_GNN_SAMPLER_UTIL_H_
