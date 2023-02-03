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

#ifndef GRIN_GRIN_CSR_EDGECUT_FRAGMENT_BASE_H_
#define GRIN_GRIN_CSR_EDGECUT_FRAGMENT_BASE_H_

#include <assert.h>

#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/immutable_csr.h"
#include "grape/vertex_map/global_vertex_map.h"

#include "grin/grin_edgecut_fragment_base.h"
#include "grin/src/predefine.h"

extern "C" {
#include "grin/include/topology/adjacentlist.h"
#include "grin/include/topology/vertexlist.h"
}

namespace grape {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename TRAITS_T>
class GRIN_CSREdgecutFragmentBase
    : public GRIN_EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T,
                                      TRAITS_T> {
 public:
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using base_t =
      GRIN_EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, TRAITS_T>;

  using base_t::fid_;
  using base_t::g_;
  using base_t::IsInnerVertex;
  using base_t::IsOuterVertex;
  using base_t::pg_;

  GRIN_CSREdgecutFragmentBase() {}

  inline size_t GetEdgeNum() const override {
    return get_edge_num(g_);
  }

  inline bool HasChild(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_adjacent_list(g_, Direction::OUT, (void*)(&v));
    return get_adjacent_list_size(al) > 0;
  }

  inline bool HasParent(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_adjacent_list(g_, Direction::IN, (void*)(&v));
    return get_adjacent_list_size(al) > 0;
  }

  inline int GetLocalOutDegree(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_adjacent_list(g_, Direction::OUT, (void*)(&v));
    return get_adjacent_list_size(al);
  }

  inline int GetLocalInDegree(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    auto al = get_adjacent_list(g_, Direction::IN, (void*)(&v));
    return get_adjacent_list_size(al);
  }

  /**
   * @brief Returns the incoming adjacent vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent vertices of v.
   *
   * @attention Only inner vertex is available.
   */
  inline adj_list_t GetIncomingAdjList(const vertex_t& v) override {
    auto al = get_adjacent_list(g_, Direction::IN, (void*)(&v));
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  /**
   * @brief Returns the incoming adjacent vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent vertices of v.
   *
   * @attention Only inner vertex is available.
   */
  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v) const override {
    auto al = get_adjacent_list(g_, Direction::IN, (void*)(&v));
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  /**
   * @brief Returns the outgoing adjacent vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent vertices of v.
   *
   * @attention Only inner vertex is available.
   */
  inline adj_list_t GetOutgoingAdjList(const vertex_t& v) override {
    auto al = get_adjacent_list(g_, Direction::OUT, (void*)(&v));
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  /**
   * @brief Returns the outgoing adjacent vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent vertices of v.
   *
   * @attention Only inner vertex is available.
   */
  inline const_adj_list_t GetOutgoingAdjList(const vertex_t& v) const override {
    auto al = get_adjacent_list(g_, Direction::OUT, (void*)(&v));
    auto aiter = static_cast<nbr_t*>(get_adjacent_list_begin(al));
    return const_adj_list_t(aiter, aiter + get_adjacent_list_size(al));
  }

  /**
   * @brief Check if inner vertex v is an incoming border vertex, that is,
   * existing edge u->v, u is an outer vertex.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is incoming border vertex, false otherwise.
   * @attention This method is only available when application set message
   * strategy as kAlongOutgoingEdgeToOuterVertex.
   */
  inline bool IsIncomingBorderVertex(const vertex_t& v) const {
    return (!idst_.empty() && IsInnerVertex(v) &&
            !idst_.is_empty(v.GetValue()));
  }

  /**
   * @brief Check if inner vertex v is an outgoing border vertex, that is,
   * existing edge v->u, u is an outer vertex.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is outgoing border vertex, false otherwise.
   * @attention This method is only available when application set message
   * strategy as kAlongIncomingEdgeToOuterVertex.
   */
  inline bool IsOutgoingBorderVertex(const vertex_t& v) const {
    return (!odst_.empty() && IsInnerVertex(v) &&
            !odst_.is_empty(v.GetValue()));
  }

  /**
   * @brief Check if inner vertex v is an border vertex, that is,
   * existing edge v->u or u->v, u is an outer vertex.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is border vertex, false otherwise.
   * @attention This method is only available when application set message
   * strategy as kAlongEdgeToOuterVertex.
   */
  inline bool IsBorderVertex(const vertex_t& v) const {
    return (!iodst_.empty() && IsInnerVertex(v) &&
            !iodst_.is_empty(v.GetValue()));
  }

  /**
   * @brief Return the incoming edge destination fragment ID list of a inner
   * vertex.
   *
   * @param v Input vertex.
   *
   * @return The incoming edge destination fragment ID list.
   *
   * @attention This method is only available when application set message
   * strategy as kAlongIncomingEdgeToOuterVertex.
   */
  inline DestList IEDests(const vertex_t& v) const override {
    assert(!idst_.empty());
    assert(IsInnerVertex(v));
    return DestList(idst_.get_begin(v.GetValue()), idst_.get_end(v.GetValue()));
  }

  /**
   * @brief Return the outgoing edge destination fragment ID list of a Vertex.
   *
   * @param v Input vertex.
   *
   * @return The outgoing edge destination fragment ID list.
   *
   * @attention This method is only available when application set message
   * strategy as kAlongOutgoingedge_toOuterVertex.
   */
  inline DestList OEDests(const vertex_t& v) const override {
    assert(!odst_.empty());
    assert(IsInnerVertex(v));
    return DestList(odst_.get_begin(v.GetValue()), odst_.get_end(v.GetValue()));
  }

  /**
   * @brief Return the edge destination fragment ID list of a inner vertex.
   *
   * @param v Input vertex.
   *
   * @return The edge destination fragment ID list.
   *
   * @attention This method is only available when application set message
   * strategy as kAlongedge_toOuterVertex.
   */
  inline DestList IOEDests(const vertex_t& v) const override {
    assert(!iodst_.empty());
    assert(IsInnerVertex(v));
    return DestList(iodst_.get_begin(v.GetValue()),
                    iodst_.get_end(v.GetValue()));
  }

  using base_t::GetFragId;
  using base_t::GetIncomingAdjList;
  using base_t::GetInnerVerticesNum;
  using base_t::GetOutgoingAdjList;

 private:
  void initDestFidList(bool in_edge, bool out_edge,
                       ImmutableCSR<VID_T, fid_t>& csr) {
    std::set<fid_t> dstset;
    ImmutableCSRStreamBuilder<VID_T, fid_t> builder;

    auto vl = get_local_vertices(pg_, fid_);
    auto viter = get_vertex_list_begin(vl);
    while (has_next_vertex_iter(vl, viter)) {
      void* vh = get_vertex_from_iter(vl, viter);
      vertex_t* v = static_cast<vertex_t*>(vh);
      dstset.clear();
      if (in_edge) {
        adj_list_t al = GetIncomingAdjList(*v);
        nbr_t* ptr = al.begin_pointer();
        nbr_t* end = al.end_pointer();
        while (ptr != end) {
          if (IsOuterVertex(ptr->neighbor)) {
            dstset.insert(GetFragId(ptr->neighbor));
          }
          ++ptr;
        }
      }
      if (out_edge) {
        adj_list_t al = GetOutgoingAdjList(*v);
        nbr_t* ptr = al.begin_pointer();
        nbr_t* end = al.end_pointer();
        while (ptr != end) {
          if (IsOuterVertex(ptr->neighbor)) {
            dstset.insert(GetFragId(ptr->neighbor));
          }
          ++ptr;
        }
      }
      builder.add_edges(dstset.begin(), dstset.end());
    }
  }

  void buildMessageDestination(const MessageStrategy& msg_strategy) {
    if (msg_strategy == MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initDestFidList(false, true, odst_);
    } else if (msg_strategy ==
               MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      initDestFidList(true, false, idst_);
    } else if (msg_strategy == MessageStrategy::kAlongEdgeToOuterVertex) {
      initDestFidList(true, true, iodst_);
    }
  }

 public:
  using base_t::initMirrorInfo;
  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) override {
    if (conf.message_strategy == MessageStrategy::kAlongEdgeToOuterVertex ||
        conf.message_strategy ==
            MessageStrategy::kAlongIncomingEdgeToOuterVertex ||
        conf.message_strategy ==
            MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      buildMessageDestination(conf.message_strategy);
    }

    if (conf.need_mirror_info) {
      initMirrorInfo(comm_spec);
    }
  }

  using base_t::Vertices;

 protected:
  ImmutableCSR<VID_T, fid_t> idst_, odst_, iodst_;
};

}  // namespace grape

#endif  // GRIN_GRIN_CSR_EDGECUT_FRAGMENT_BASE_H_
