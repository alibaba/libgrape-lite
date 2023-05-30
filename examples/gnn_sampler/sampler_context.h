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

#ifndef EXAMPLES_GNN_SAMPLER_SAMPLER_CONTEXT_H_
#define EXAMPLES_GNN_SAMPLER_SAMPLER_CONTEXT_H_

#include <string>
#include <unordered_map>
#include <vector>

#include <grape/app/void_context.h>
#include <grape/grape.h>
#include <grape/io/line_parser_base.h>

#include "flat_hash_map/flat_hash_map.hpp"
#include "util.h"
#include "xoroshiro/xoroshiro.hpp"

namespace grape {
template <typename FRAG_T>
class SamplerContext : public grape::VoidContext<FRAG_T> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;

 public:
  explicit SamplerContext(const fragment_t& fragment)
      : grape::VoidContext<FRAG_T>(fragment) {}

  void Init(ParallelMessageManager& messages, const std::string& strategy,
            const std::string& sampler_hop_and_num,
            const std::vector<std::string>& queries) {
    auto& frag = this->fragment();
#ifdef PROFILING
    time_init -= GetCurrentTime();
#endif
    if (strategy == "random") {
      random_strategy = RandomStrategy::Random;
    } else if (strategy == "edge_weight") {
      random_strategy = RandomStrategy::EdgeWeight;
    } else if (strategy == "top_k") {
      random_strategy = RandomStrategy::TopK;
    } else {
      LOG(FATAL) << "Invalid strategy";
    }

    parse_hop_and_num(sampler_hop_and_num, nums_of_hop, hop_size);

    if (!queries.empty()) {
      oid_t oid;
      vid_t gid;
      for (auto& v : queries) {
        grape::internal::match<oid_t>(v.c_str(), oid, nullptr);
        vertex_t vert;
        if (frag.GetInnerVertex(oid, vert)) {
          gid = frag.GetInnerVertexGid(vert);
          if (random_result.find(gid) == random_result.end()) {
            random_result[gid].resize(hop_size.back(), 0);
          }
        }
      }
    } else {
      auto inner_vertices = frag.InnerVertices();
      for (auto& v : inner_vertices) {
        random_result[frag.Vertex2Gid(v)].resize(hop_size.back(), 0);
      }
    }
#ifdef PROFILING
    time_init += GetCurrentTime();
    LOG(INFO) << "Time Init: " << time_init;
#endif
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto t_begin = grape::GetCurrentTime();

    for (auto& it : random_result) {
      std::stringstream ss;
      ss << frag.Gid2Oid(it.first);
      for (auto gid : it.second) {
        ss << " " << frag.Gid2Oid(gid);
      }
      os << ss.str() << std::flush;
    }

    auto elapsed = grape::GetCurrentTime() - t_begin;
    LOG(INFO) << "Output time: " << elapsed << " s";
  }

  using message_t = std::tuple<vid_t, vid_t, uint32_t>;

  uint32_t cur_hop;
  std::vector<std::shared_ptr<xoroshiro128plus64>> rngs;
  RandomStrategy random_strategy;
  std::vector<uint32_t> nums_of_hop;
  std::vector<uint32_t> hop_size;
  std::shared_ptr<std::vector<std::vector<message_t>>> random_cache;
  ska::flat_hash_map<vid_t, std::vector<vid_t>> random_result;
#ifdef PROFILING
  double time_init = 0;
  double time_pval = 0;
  double time_inceval = 0;
  double time_inceval_get_apply_msg = 0;
  double time_inceval_gen_send_msg = 0;
#endif
};

}  // namespace grape

#endif  // EXAMPLES_GNN_SAMPLER_SAMPLER_CONTEXT_H_
