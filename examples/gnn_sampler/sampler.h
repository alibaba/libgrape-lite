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

#ifndef EXAMPLES_GNN_SAMPLER_SAMPLER_H_
#define EXAMPLES_GNN_SAMPLER_SAMPLER_H_

#include <iomanip>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <grape/parallel/parallel_engine.h>
#include <grape/utils/concurrent_queue.h>

#include "flat_hash_map/flat_hash_map.hpp"
#include "fragment_indices.h"
#include "sampler_context.h"

namespace grape {

template <typename FRAG_T>
class Sampler : public ParallelAppBase<FRAG_T, SamplerContext<FRAG_T>>,
                public ParallelEngine {
  INSTALL_PARALLEL_WORKER(Sampler<FRAG_T>, SamplerContext<FRAG_T>, FRAG_T);
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = grape::Vertex<vid_t>;
  using message_t = std::tuple<vid_t, vid_t, uint32_t>;
  using rng_ptr_t = std::shared_ptr<xoroshiro128plus64>;
  using result_entry_t =
      ska::detailv3::sherwood_v3_entry<std::pair<vid_t, std::vector<vid_t>>>;
  using thread_local_message_buffer_t =
      ThreadLocalMessageBuffer<message_manager_t>;

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto& random_result = ctx.random_result;
    auto& cur_hop = ctx.cur_hop;
    cur_hop = 1;

#ifdef PROFILING
    ctx.time_pval -= GetCurrentTime();
#endif
    LOG(INFO) << "Sampler - Thread Num: " << thread_num();

    messages.InitChannels(thread_num(), 2 * 1023 * 64, 2 * 1024 * 64);
    ctx.rngs.resize(thread_num());

    for (size_t idx = 0; idx < thread_num(); ++idx) {
      uint64_t s0 = rand() % 1024 + 1;  // NOLINT(runtime/threadsafe_fn)
      uint64_t s1 = rand() % 1024 + 1;  // NOLINT(runtime/threadsafe_fn)
      ctx.rngs[idx] = std::make_shared<xoroshiro128plus64>(s0, s1);
    }

    auto for_caches =
        std::make_shared<std::vector<std::vector<message_t>>>(thread_num());

    auto begin = random_result.begin().current,
         end = random_result.end().current;

    ForEach(begin, end,
            [this, &ctx, &frag, &messages, &for_caches, cur_hop](
                int tid, const result_entry_t& entry) {
              if (entry.has_value()) {          // skip empty cells
                auto root = entry.value.first;  // root is represented by gid
                RandomSelect(frag, ctx, messages.Channels()[tid], ctx.rngs[tid],
                             root, root, 0, cur_hop, (*for_caches)[tid]);
              }
            });

    ctx.random_cache = for_caches;
    messages.ForceContinue();
#ifdef PROFILING
    ctx.time_pval += GetCurrentTime();
    LOG(INFO) << "Time PEval: " << ctx.time_pval;
#endif
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto& cur_hop = ctx.cur_hop;
    ++cur_hop;
    auto& random_cache = ctx.random_cache;
#ifdef PROFILING
    ctx.time_inceval -= GetCurrentTime();
    ctx.time_inceval_get_apply_msg -= GetCurrentTime();
#endif
    auto for_caches =
        std::make_shared<std::vector<std::vector<message_t>>>(thread_num());

    messages.ParallelProcess<message_t>(
        thread_num(), [this, &frag, &ctx, &messages, cur_hop, &for_caches](
                          int tid, message_t& msg) {
          vid_t root = std::get<0>(msg);
          vid_t gid = std::get<1>(msg);
          vid_t pos = std::get<2>(msg);

          if (pos < ctx.hop_size[cur_hop - 1]) {
            ctx.random_result[root][pos] = gid;
          } else {
            RandomSelect(frag, ctx, messages.Channels()[tid], ctx.rngs[tid],
                         gid, root, pos, cur_hop, (*for_caches)[tid]);
          }
        });
#ifdef PROFILING
    ctx.time_inceval_get_apply_msg += GetCurrentTime();
    ctx.time_inceval_gen_send_msg -= GetCurrentTime();
#endif

    // process random_cache
    auto begin = random_cache->begin(), end = random_cache->end();
    ForEach(begin, end,
            [this, &ctx, &frag, &messages, &for_caches, cur_hop](
                int tid, const std::vector<message_t>& msgs) {
              for (auto const& msg : msgs) {
                vid_t root = std::get<0>(msg);
                vid_t u_id = std::get<1>(msg);
                vid_t pos = std::get<2>(msg);
                RandomSelect(frag, ctx, messages.Channels()[tid], ctx.rngs[tid],
                             u_id, root, pos, cur_hop, (*for_caches)[tid]);
              }
            });

    if (cur_hop <= ctx.nums_of_hop.size()) {
      ctx.random_cache = for_caches;
      messages.ForceContinue();
    }
#ifdef PROFILING
    ctx.time_inceval_gen_send_msg += GetCurrentTime();
    ctx.time_inceval += GetCurrentTime();
    LOG(INFO) << "Time IncEval: " << ctx.time_inceval;
    LOG(INFO) << "Time IncEval GetApply Msg: "
              << ctx.time_inceval_get_apply_msg;
    LOG(INFO) << "Time IncEval Gen&Send Msg: " << ctx.time_inceval_gen_send_msg;
#endif
  }

 private:
  void RandomSelect(const fragment_t& frag, context_t& ctx,
                    thread_local_message_buffer_t& channel,
                    rng_ptr_t& random_engine, vid_t u_id, vid_t root,
                    const uint32_t pos, const uint32_t cur_hop,
                    std::vector<message_t>& random_cache) {
    vertex_t n;
    auto& nums_of_hop = ctx.nums_of_hop;
    uint32_t hop_num = nums_of_hop.size();
    uint32_t random_num = (cur_hop <= hop_num) ? nums_of_hop[cur_hop - 1] : 0;
    uint32_t num_of_next_hop = (cur_hop < hop_num) ? nums_of_hop[cur_hop] : 0;
    auto weight_indices =
        dynamic_cast<WeightIndices<oid_t, vid_t, vdata_t, edata_t>*>(
            frag.GetFragmentIndices().get());

    frag.Gid2Vertex(u_id, n);

    if (ctx.random_strategy == RandomStrategy::Random) {
      auto& ids = weight_indices->GetOrderedIds(n);
      for (uint32_t count = 0; count < random_num; ++count) {
        uint64_t index = (*random_engine)() % ids.size();
        auto v_id = ids[index];
        PushResult(frag, ctx, channel, root, v_id, hop_num, num_of_next_hop,
                   pos, random_cache, cur_hop, count);
      }
    } else if (ctx.random_strategy == RandomStrategy::EdgeWeight) {
      auto& acc_weights = weight_indices->GetAccWeights(n);
      auto& ordered_ids = weight_indices->GetOrderedIds(n);
      for (uint32_t count = 0; count < random_num; ++count) {
        edata_t linear =
            (*random_engine)() * acc_weights.back() /
            static_cast<edata_t>(std::numeric_limits<uint64_t>::max());
        auto it =
            std::lower_bound(acc_weights.begin(), acc_weights.end(), linear);
        auto v_id = ordered_ids[std::distance(acc_weights.begin(), it)];
        PushResult(frag, ctx, channel, root, v_id, hop_num, num_of_next_hop,
                   pos, random_cache, cur_hop, count);
      }
    } else if (ctx.random_strategy == RandomStrategy::TopK) {
      auto& ids = weight_indices->GetOrderedIds(n);
      int i = ids.size() - 1;
      for (uint32_t count = 0; count < random_num; ++count) {
        if (i < 0) {
          i = ids.size() - 1;  // recycle
        }
        auto v_id = ids[i--];
        PushResult(frag, ctx, channel, root, v_id, hop_num, num_of_next_hop,
                   pos, random_cache, cur_hop, count);
      }
    }
  }

  void PushResult(const fragment_t& frag, context_t& ctx,
                  thread_local_message_buffer_t& channel, const vid_t root,
                  const vid_t v_gid, const uint32_t hop_num,
                  const uint32_t num_of_next_hop, const uint32_t pos,
                  std::vector<message_t>& random_cache, const uint32_t cur_hop,
                  const uint32_t count) {
    auto& result = ctx.random_result;
    auto& hop_size = ctx.hop_size;

    if (frag.GetFragIdByGid(root) == frag.fid()) {
      result[root][count + pos] = v_gid;
    } else {
      channel.SendToFragment(frag.GetFragIdByGid(root),
                             std::make_tuple(root, v_gid, pos + count));
    }

    if (cur_hop < hop_num) {
      uint32_t next_pos =
          (pos + count - hop_size[cur_hop - 1]) * num_of_next_hop +
          hop_size[cur_hop];
      auto v_fid = frag.GetFragIdByGid(v_gid);

      if (v_fid == frag.fid()) {
        random_cache.emplace_back(root, v_gid, next_pos);
      } else {
        channel.SendToFragment(v_fid, std::make_tuple(root, v_gid, next_pos));
      }
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_GNN_SAMPLER_SAMPLER_H_
