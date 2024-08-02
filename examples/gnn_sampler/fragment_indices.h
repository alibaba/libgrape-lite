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

#ifndef EXAMPLES_GNN_SAMPLER_FRAGMENT_INDICES_H_
#define EXAMPLES_GNN_SAMPLER_FRAGMENT_INDICES_H_

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <grape/fragment/edgecut_fragment_base.h>
#include <grape/util.h>
#include <grape/utils/gcontainer.h>
#include <grape/utils/vertex_array.h>

#include "append_only_edgecut_fragment.h"
#include "flat_hash_map/flat_hash_map.hpp"

namespace grape {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
struct AppendOnlyEdgecutFragmentTraits;

}  // namespace grape

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class FragmentIndicesBase {
 public:
  virtual ~FragmentIndicesBase() {}

  static std::unique_ptr<FragmentIndicesBase<OID_T, VID_T, VDATA_T, EDATA_T>>
  Create();

  virtual void
  Init(grape::EdgecutFragmentBase<
       OID_T, VID_T, VDATA_T, EDATA_T,
       grape::AppendOnlyEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T>>*
           frag) = 0;

  virtual void Insert(VID_T u, VID_T v, EDATA_T data) = 0;

  virtual void Rebuild() = 0;

  virtual void Resize(size_t new_size) = 0;
};

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class WeightIndices final
    : public FragmentIndicesBase<OID_T, VID_T, VDATA_T, EDATA_T> {
 public:
  using vertex_t = grape::Vertex<VID_T>;
  using oid_t = OID_T;
  using vid_t = VID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;

  struct WeightCollections {
    ska::flat_hash_map<vid_t, edata_t> weights;
    std::set<std::pair<edata_t, vid_t>> ordered_weights;
    std::vector<edata_t> acc_weights;
    std::vector<vid_t> ordered_ids;
  };

  void
  Init(grape::EdgecutFragmentBase<
       oid_t, vid_t, vdata_t, edata_t,
       grape::AppendOnlyEdgecutFragmentTraits<oid_t, vid_t, vdata_t, edata_t>>*
           frag) override {
    weight_indices_.resize(frag->GetInnerVerticesNum());
    for (auto& v : frag->InnerVertices()) {
      int neighbor_num = frag->GetLocalOutDegree(v);
      vid_t lid = v.GetValue();

      auto& weights = weight_indices_[lid].weights;
      auto& ordered_weights = weight_indices_[lid].ordered_weights;
      auto& acc_weights = weight_indices_[lid].acc_weights;
      auto& ordered_ids = weight_indices_[lid].ordered_ids;

      weights.reserve(neighbor_num);

      auto oes = frag->GetOutgoingAdjList(v);
      for (auto e = oes.begin(); e != oes.end(); ++e) {
        auto u_vid = frag->Vertex2Gid(e->neighbor);
        weights.emplace(u_vid, e->data);
      }

      for (auto const& kv : weights) {
        ordered_weights.emplace(kv.second, kv.first);
      }

      acc_weights.reserve(ordered_weights.size());
      ordered_ids.reserve(ordered_weights.size());
      edata_t acc_weight = 0;
      for (auto const& kv : ordered_weights) {
        acc_weight += kv.first;
        acc_weights.emplace_back(acc_weight);
        ordered_ids.emplace_back(kv.second);
      }
    }
  }

  void Insert(vid_t lid, vid_t nei_id, edata_t w) override {
    auto& weights = weight_indices_[lid].weights;
    auto& ordered_weights = weight_indices_[lid].ordered_weights;

    auto iter = weights.find(nei_id);
    if (iter != weights.end()) {
      // update index
      edata_t pw = iter->second;
      weights[nei_id] = w;
      ordered_weights.erase(std::make_pair(pw, iter->first));
      ordered_weights.emplace(w, nei_id);
    } else {
      // insert index
      weights.emplace(nei_id, w);
      ordered_weights.emplace(w, nei_id);
    }
    affected_ids_.emplace(lid);
  }

  const ska::flat_hash_map<vid_t, edata_t>& GetWeights(
      const vertex_t& v) const {
    return weight_indices_[v.GetValue()].weights;
  }

  const std::set<std::pair<edata_t, vid_t>>& GetOrderedWeights(
      const vertex_t& v) const {
    return weight_indices_[v.GetValue()].ordered_weights;
  }

  const std::vector<edata_t>& GetAccWeights(const vertex_t& v) const {
    return weight_indices_[v.GetValue()].acc_weights;
  }

  const std::vector<vid_t>& GetOrderedIds(const vertex_t& v) const {
    return weight_indices_[v.GetValue()].ordered_ids;
  }

  void Rebuild() override {
    for (vid_t vid : affected_ids_) {
      auto& ordered_weights = weight_indices_[vid].ordered_weights;
      auto& acc_weights = weight_indices_[vid].acc_weights;
      auto& ordered_ids = weight_indices_[vid].ordered_ids;

      acc_weights.resize(ordered_weights.size());
      ordered_ids.resize(ordered_weights.size());
      edata_t acc_weight = 0.0;
      size_t idx = 0;
      for (auto const& kv : ordered_weights) {
        acc_weight += kv.first;
        acc_weights[idx] = acc_weight;
        ordered_ids[idx++] = kv.second;
      }
    }
    affected_ids_.clear();
  }

  void Resize(size_t new_size) override { weight_indices_.resize(new_size); }

 private:
  std::set<vid_t> affected_ids_;
  std::vector<WeightCollections> weight_indices_;
};

template <typename OID_T, typename VID_T, typename VDATA_T>
class WeightIndices<OID_T, VID_T, VDATA_T, grape::EmptyType>
    : public FragmentIndicesBase<OID_T, VID_T, VDATA_T, grape::EmptyType> {
 public:
  void Init(
      grape::EdgecutFragmentBase<OID_T, VID_T, VDATA_T, grape::EmptyType,
                                 grape::AppendOnlyEdgecutFragmentTraits<
                                     OID_T, VID_T, VDATA_T, grape::EmptyType>>*
          frag) override {}

  void Insert(VID_T vid, VID_T nei_id, grape::EmptyType w) override {}

  void Rebuild() override {}

  void Resize(size_t new_size) override {}
};

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
std::unique_ptr<FragmentIndicesBase<OID_T, VID_T, VDATA_T, EDATA_T>>
FragmentIndicesBase<OID_T, VID_T, VDATA_T, EDATA_T>::Create() {
  return std::unique_ptr<FragmentIndicesBase<OID_T, VID_T, VDATA_T, EDATA_T>>(
      new WeightIndices<OID_T, VID_T, VDATA_T, EDATA_T>());
}

#endif  // EXAMPLES_GNN_SAMPLER_FRAGMENT_INDICES_H_
