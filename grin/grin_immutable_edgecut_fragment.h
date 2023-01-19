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

#ifndef GRIN_GRIN_IMMUTABLE_EDGECUT_FRAGMENT_H_
#define GRIN_GRIN_IMMUTABLE_EDGECUT_FRAGMENT_H_

#include <assert.h>
#include <stddef.h>

#include <algorithm>
#include <iosfwd>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "grape/config.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/immutable_csr.h"
#include "grape/graph/vertex.h"
#include "grape/types.h"
#include "grape/util.h"
#include "grape/utils/vertex_array.h"
#include "grape/vertex_map/global_vertex_map.h"
#include "grape/worker/comm_spec.h"

#include "grin/grin_csr_edgecut_fragment_base.h"
#include "grin/include/predefine.h"

extern "C" {
#include "grin/include/partition/partition.h"
#include "grin/include/topology/structure.h"
}

namespace grape {
class CommSpec;

/**
 * @brief GRIN_ImmutableEdgecutFragment is the GRIN version 
 * of ImmutableEdgecutFragment in grape,
 * which is a kind of edgecut fragment.
 *
 * @tparam OID_T Type of original ID.
 * @tparam VID_T Type of global ID and local ID.
 * @tparam VDATA_T Type of data on vertices.
 * @tparam EDATA_T Type of data on edges.
 * @tparam LoadStrategy The strategy to store adjacency information, default is
 * only_out.
 *
 * With an edgecut partition, each vertex is assigned to a fragment.
 * In a fragment, inner vertices are those vertices assigned to it, and the
 * outer vertices are the remaining vertices adjacent to some of the inner
 * vertices. The load strategy defines how to store the adjacency between inner
 * and outer vertices.
 *
 * For example, a graph
 * G = {V, E}
 * V = {v0, v1, v2, v3, v4}
 * E = {(v0, v2), (v0, v3), (v1, v0), (v3, v1), (v3, v4), (v4, v1), (v4, v2)}
 *
 * Subset V_0 = {v0, v1} is assigned to fragment_0, so InnerVertices_0 = {v0,
 * v1}
 *
 * If the load strategy is kOnlyIn:
 * All incoming edges (along with the source vertices) of inner vertices will be
 * stored in a fragment. So,
 * OuterVertices_0 = {v3, v4}, E_0 = {(v1, v0), (v3, v1), (v4, v1)}
 *
 * If the load strategy is kOnlyOut:
 * All outgoing edges (along with the destination vertices) of inner vertices
 * will be stored in a fragment. So,
 * OuterVertices_0 = {v2, v3}, E_0 = {(v0, v2), (v0, v3), (v1, v0)}
 *
 * If the load strategy is kBothOutIn:
 * All incoming edges (along with the source vertices) and outgoing edges (along
 * with destination vertices) of inner vertices will be stored in a fragment.
 * So, OuterVertices_0 = {v2, v3, v4}, E_0 = {(v0, v2), (v0, v3), (v1, v0), (v3,
 * v1), (v4, v1), (v4, v2)}
 *
 * Inner vertices and outer vertices of a fragment will be given a local ID
 * {0, 1, ..., ivnum - 1, ivnum, ..., ivnum + ovnum - 1},
 * then iterate on vertices can be implemented to increment the local ID.
 * Also, the sets of inner vertices, outer vertices and all vertices are ranges
 * of local ID.
 *
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy _load_strategy = LoadStrategy::kOnlyOut,
          typename VERTEX_MAP_T = GlobalVertexMap<OID_T, VID_T>>
class GRIN_ImmutableEdgecutFragment
    : public GRIN_CSREdgecutFragmentBase<
          OID_T, VID_T, VDATA_T, EDATA_T,
          ImmutableEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T,
                                         VERTEX_MAP_T>> {
 public:
  using traits_t = ImmutableEdgecutFragmentTraits<OID_T, VID_T, VDATA_T,
                                                  EDATA_T, VERTEX_MAP_T>;
  using base_t =
      GRIN_CSREdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_map_t = typename traits_t::vertex_map_t;

  static constexpr LoadStrategy load_strategy = _load_strategy;

  using vertex_range_t = VertexRange<VID_T>;
  using inner_vertices_t = typename traits_t::inner_vertices_t;
  using outer_vertices_t = typename traits_t::outer_vertices_t;
  using vertices_t = typename traits_t::vertices_t;

  template <typename T>
  using inner_vertex_array_t = VertexArray<inner_vertices_t, T>;

  template <typename T>
  using outer_vertex_array_t = VertexArray<outer_vertices_t, T>;

  template <typename T>
  using vertex_array_t = VertexArray<vertices_t, T>;

  GRIN_ImmutableEdgecutFragment() {}

  explicit GRIN_ImmutableEdgecutFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : GRIN_FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>(vm_ptr) {}

  virtual ~GRIN_ImmutableEdgecutFragment() = default;

  using base_t::init;
  using base_t::SetVertexMap;

  void Init(const PartitionedGraph g, std::shared_ptr<vertex_map_t> vm_ptr) {
    init(g);
    SetVertexMap(vm_ptr);
  }

  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) override {
    base_t::PrepareToRunApp(comm_spec, conf);
  }

  using base_t::fid_;
  using base_t::g_;
  using base_t::pg_;

  using base_t::GetFragId;
  using base_t::IsInnerVertex;

  inline const VDATA_T GetData(const vertex_t& v) const override {
    return get_vertex_data_value(g_, v.GetValue());
  }

  bool OuterVertexGid2Lid(VID_T gid, VID_T& lid) const override {
    std::stringstream ss;
    ss << gid;
    lid = get_vertex_from_deserialization(pg_, fid_, ss.str().c_str());
    return lid != NULL_VERTEX;
  }

  VID_T GetOuterVertexGid(vertex_t v) const override {
    return get_master_vertex_for_vertex(pg_, fid_, v.GetValue());
  }

 public:
  using base_t::GetIncomingAdjList;
  using base_t::GetOutgoingAdjList;

  using adj_list_t = typename base_t::adj_list_t;
  using const_adj_list_t = typename base_t::const_adj_list_t;
  inline adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    auto al = get_local_adjacent_list(pg_, Direction::IN, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline const_adj_list_t GetIncomingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_local_adjacent_list(pg_, Direction::IN, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list(pg_, Direction::IN, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline const_adj_list_t GetIncomingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list(pg_, Direction::IN, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    auto al = get_local_adjacent_list(pg_, Direction::OUT, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline const_adj_list_t GetOutgoingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_local_adjacent_list(pg_, Direction::OUT, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list(pg_, Direction::OUT, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline const_adj_list_t GetOutgoingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list(pg_, Direction::OUT, fid_, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v,
                                       fid_t src_fid) override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list_by_partition(pg_, Direction::IN, src_fid,
                                                    v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                             fid_t src_fid) const override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list_by_partition(pg_, Direction::IN, src_fid,
                                                    v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                       fid_t dst_fid) override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list_by_partition(pg_, Direction::OUT,
                                                    dst_fid, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  inline const_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                             fid_t dst_fid) const override {
    assert(IsInnerVertex(v));
    auto al = get_remote_adjacent_list_by_partition(pg_, Direction::OUT,
                                                    dst_fid, v.GetValue());
    assert(al != NULL_LIST);
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }
};

}  // namespace grape

#endif  // GRIN_GRIN_IMMUTABLE_EDGECUT_FRAGMENT_H_
