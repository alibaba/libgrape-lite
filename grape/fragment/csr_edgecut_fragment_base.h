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

#ifndef GRAPE_FRAGMENT_CSR_EDGECUT_FRAGMENT_BASE_H_
#define GRAPE_FRAGMENT_CSR_EDGECUT_FRAGMENT_BASE_H_

#include <assert.h>

#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/immutable_csr.h"
#include "grape/vertex_map/global_vertex_map.h"

namespace grape {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename TRAITS_T>
class CSREdgecutFragmentBase
    : public EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, TRAITS_T> {
 public:
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using base_t = EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, TRAITS_T>;

  using base_t::IsInnerVertex;
  using base_t::IsOuterVertex;

  CSREdgecutFragmentBase() {}

  inline size_t GetEdgeNum() const override {
    return ie_.edge_num() + oe_.edge_num();
  }

  inline bool HasChild(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return !oe_.is_empty(v.GetValue());
  }

  inline bool HasParent(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return !ie_.is_empty(v.GetValue());
  }

  inline int GetLocalOutDegree(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return oe_.degree(v.GetValue());
  }

  inline int GetLocalInDegree(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return ie_.degree(v.GetValue());
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
    return adj_list_t(ie_.get_begin(v.GetValue()), ie_.get_end(v.GetValue()));
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
    return const_adj_list_t(ie_.get_begin(v.GetValue()),
                            ie_.get_end(v.GetValue()));
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
    return adj_list_t(oe_.get_begin(v.GetValue()), oe_.get_end(v.GetValue()));
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
    return const_adj_list_t(oe_.get_begin(v.GetValue()),
                            oe_.get_end(v.GetValue()));
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
  using base_t::GetInnerVerticesNum;
  using base_t::GetIncomingAdjList;
  using base_t::GetOutgoingAdjList;

 private:
  using csr_t = typename TRAITS_T::csr_t;
  using csr_builder_t = typename TRAITS_T::csr_builder_t;

  void initDestFidList(bool in_edge, bool out_edge,
                       ImmutableCSR<VID_T, fid_t>& csr) {
    std::set<fid_t> dstset;
    ImmutableCSRStreamBuilder<VID_T, fid_t> builder;

    VID_T ivnum = GetInnerVerticesNum();
    for (VID_T i = 0; i < ivnum; ++i) {
      dstset.clear();
      if (in_edge) {
        nbr_t* ptr = ie_.get_begin(i);
        nbr_t* end = ie_.get_end(i);
        while (ptr != end) {
          if (IsOuterVertex(ptr->neighbor)) {
            dstset.insert(GetFragId(ptr->neighbor));
          }
          ++ptr;
        }
      }
      if (out_edge) {
        nbr_t* ptr = oe_.get_begin(i);
        nbr_t* end = oe_.get_end(i);
        while (ptr != end) {
          if (IsOuterVertex(ptr->neighbor)) {
            dstset.insert(GetFragId(ptr->neighbor));
          }
          ++ptr;
        }
      }
      builder.add_edges(dstset.begin(), dstset.end());
    }

    builder.finish(csr);
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
  using base_t::directed_;
  using base_t::Gid2Lid;
  using base_t::InnerVertexGid2Lid;
  using base_t::IsInnerVertexGid;
  using base_t::IsInnerVertexLid;
  using base_t::OuterVertexGid2Lid;
  void buildCSR(const typename csr_builder_t::vertex_range_t& vertex_range,
                std::vector<Edge<VID_T, EDATA_T>>& edges,
                LoadStrategy load_strategy) {
    csr_builder_t ie_builder, oe_builder;
    ie_builder.init(vertex_range);
    oe_builder.init(vertex_range);

    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    auto parse_iter_in = [&](Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        if (IsInnerVertexGid(e.src)) {
          InnerVertexGid2Lid(e.src, e.src);
        } else {
          CHECK(OuterVertexGid2Lid(e.src, e.src));
          oe_builder.inc_degree(e.src);
        }
        InnerVertexGid2Lid(e.dst, e.dst);
        ie_builder.inc_degree(e.dst);
      }
    };
    auto parse_iter_out = [&](Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        InnerVertexGid2Lid(e.src, e.src);
        oe_builder.inc_degree(e.src);
        if (IsInnerVertexGid(e.dst)) {
          InnerVertexGid2Lid(e.dst, e.dst);
        } else {
          CHECK(OuterVertexGid2Lid(e.dst, e.dst));
          ie_builder.inc_degree(e.dst);
        }
      }
    };
    auto parse_iter_out_in = [&](Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        Gid2Lid(e.src, e.src);
        oe_builder.inc_degree(e.src);
        Gid2Lid(e.dst, e.dst);
        ie_builder.inc_degree(e.dst);
      }
    };
    auto parse_iter_in_undirected = [&](Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        if (IsInnerVertexGid(e.src)) {
          InnerVertexGid2Lid(e.src, e.src);
          ie_builder.inc_degree(e.src);
        } else {
          CHECK(OuterVertexGid2Lid(e.src, e.src));
          oe_builder.inc_degree(e.src);
        }
        if (IsInnerVertexGid(e.dst)) {
          InnerVertexGid2Lid(e.dst, e.dst);
          ie_builder.inc_degree(e.dst);
        } else {
          CHECK(OuterVertexGid2Lid(e.dst, e.dst));
          oe_builder.inc_degree(e.dst);
        }
      }
    };
    auto parse_iter_out_undirected = [&](Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        if (IsInnerVertexGid(e.src)) {
          InnerVertexGid2Lid(e.src, e.src);
          oe_builder.inc_degree(e.src);
        } else {
          CHECK(OuterVertexGid2Lid(e.src, e.src));
          ie_builder.inc_degree(e.src);
        }
        if (IsInnerVertexGid(e.dst)) {
          InnerVertexGid2Lid(e.dst, e.dst);
          oe_builder.inc_degree(e.dst);
        } else {
          CHECK(OuterVertexGid2Lid(e.dst, e.dst));
          ie_builder.inc_degree(e.dst);
        }
      }
    };
    auto parse_iter_out_in_undirected = [&](Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        Gid2Lid(e.src, e.src);
        oe_builder.inc_degree(e.src);
        ie_builder.inc_degree(e.src);
        Gid2Lid(e.dst, e.dst);
        oe_builder.inc_degree(e.dst);
        ie_builder.inc_degree(e.dst);
      }
    };
    if (load_strategy == LoadStrategy::kOnlyIn) {
      if (this->directed_) {
        for (auto& e : edges) {
          parse_iter_in(e);
        }
      } else {
        for (auto& e : edges) {
          parse_iter_in_undirected(e);
        }
      }
    } else if (load_strategy == LoadStrategy::kOnlyOut) {
      if (this->directed_) {
        for (auto& e : edges) {
          parse_iter_out(e);
        }
      } else {
        for (auto& e : edges) {
          parse_iter_out_undirected(e);
        }
      }
    } else if (load_strategy == LoadStrategy::kBothOutIn) {
      if (this->directed_) {
        for (auto& e : edges) {
          parse_iter_out_in(e);
        }
      } else {
        for (auto& e : edges) {
          parse_iter_out_in_undirected(e);
        }
      }
    } else {
      LOG(FATAL) << "Invalid load strategy";
    }

    ie_builder.build_offsets();
    oe_builder.build_offsets();

    auto insert_iter_in = [&](const Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        ie_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        if (!IsInnerVertexLid(e.src)) {
          oe_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        }
      }
    };
    auto insert_iter_out = [&](const Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        oe_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        if (!IsInnerVertexLid(e.dst)) {
          ie_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        }
      }
    };
    auto insert_iter_out_in = [&](const Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        ie_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        oe_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
      }
    };
    auto insert_iter_in_undirected = [&](const Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        if (IsInnerVertexLid(e.src)) {
          ie_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        } else {
          oe_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        }
        if (IsInnerVertexLid(e.dst)) {
          ie_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        } else {
          oe_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        }
      }
    };
    auto insert_iter_out_undirected = [&](const Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        if (IsInnerVertexLid(e.src)) {
          oe_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        } else {
          ie_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        }
        if (IsInnerVertexLid(e.dst)) {
          oe_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        } else {
          ie_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        }
      }
    };
    auto insert_iter_out_in_undirected = [&](const Edge<VID_T, EDATA_T>& e) {
      if (e.src != invalid_vid) {
        ie_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
        ie_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        oe_builder.add_edge(e.src, nbr_t(e.dst, e.edata));
        oe_builder.add_edge(e.dst, nbr_t(e.src, e.edata));
      }
    };

    if (load_strategy == LoadStrategy::kOnlyIn) {
      if (this->directed_) {
        for (auto& e : edges) {
          insert_iter_in(e);
        }
      } else {
        for (auto& e : edges) {
          insert_iter_in_undirected(e);
        }
      }
    } else if (load_strategy == LoadStrategy::kOnlyOut) {
      if (this->directed_) {
        for (auto& e : edges) {
          insert_iter_out(e);
        }
      } else {
        for (auto& e : edges) {
          insert_iter_out_undirected(e);
        }
      }
    } else if (load_strategy == LoadStrategy::kBothOutIn) {
      if (this->directed_) {
        for (auto& e : edges) {
          insert_iter_out_in(e);
        }
      } else {
        for (auto& e : edges) {
          insert_iter_out_in_undirected(e);
        }
      }
    } else {
      LOG(FATAL) << "Invalid load strategy";
    }

    ie_builder.finish(ie_);
    oe_builder.finish(oe_);
  }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    base_t::serialize(writer);
    ie_.template Serialize<IOADAPTOR_T>(writer);
    oe_.template Serialize<IOADAPTOR_T>(writer);
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    base_t::deserialize(reader);
    ie_.template Deserialize<IOADAPTOR_T>(reader);
    oe_.template Deserialize<IOADAPTOR_T>(reader);
  }

  nbr_t* get_ie_begin(const vertex_t& v) { return ie_.get_begin(v.GetValue()); }

  nbr_t* get_ie_end(const vertex_t& v) { return ie_.get_end(v.GetValue()); }

  const nbr_t* get_ie_begin(const vertex_t& v) const {
    return ie_.get_begin(v.GetValue());
  }

  const nbr_t* get_ie_end(const vertex_t& v) const {
    return ie_.get_end(v.GetValue());
  }

  nbr_t* get_oe_begin(const vertex_t& v) { return oe_.get_begin(v.GetValue()); }

  nbr_t* get_oe_end(const vertex_t& v) { return oe_.get_end(v.GetValue()); }

  const nbr_t* get_oe_begin(const vertex_t& v) const {
    return oe_.get_begin(v.GetValue());
  }

  const nbr_t* get_oe_end(const vertex_t& v) const {
    return oe_.get_end(v.GetValue());
  }

  csr_t ie_, oe_;
  ImmutableCSR<VID_T, fid_t> idst_, odst_, iodst_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_CSR_EDGECUT_FRAGMENT_BASE_H_
