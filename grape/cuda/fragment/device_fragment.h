/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef GRAPE_CUDA_FRAGMENT_DEVICE_FRAGMENT_H_
#define GRAPE_CUDA_FRAGMENT_DEVICE_FRAGMENT_H_
#include "cuda_hashmap/hash_map.h"
#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/vertex_map/device_vertex_map.h"
#include "grape/fragment/id_parser.h"
#include "grape/graph/adj_list.h"
#include "grape/types.h"
#include "grape/utils/vertex_array.h"

namespace grape {
namespace cuda {
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          grape::LoadStrategy _load_strategy, typename VERTEX_MAP_T>
class HostFragment;

namespace dev {
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          grape::LoadStrategy _load_strategy = grape::LoadStrategy::kOnlyOut>
class DeviceFragment {
 public:
  using vertex_t = Vertex<VID_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_range_t = VertexRange<VID_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;

  DEV_HOST_INLINE vertex_range_t Vertices() const {
    return vertex_range_t(0, tvnum_);
  }

  DEV_HOST_INLINE vertex_range_t InnerVertices() const {
    return vertex_range_t(0, ivnum_);
  }

  DEV_HOST_INLINE vertex_range_t OuterVertices() const {
    return vertex_range_t(ivnum_, tvnum_);
  }

  DEV_INLINE vertex_range_t OuterVertices(fid_t fid) const {
    return outer_vertices_of_frag_[fid];
  }

  DEV_INLINE ArrayView<vertex_t> MirrorVertices(fid_t fid) const {
    return mirrors_of_frag_[fid];
  }

  DEV_INLINE bool GetVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;
    if (vm_.GetGid(oid, gid)) {
      return id_parser_.get_fragment_id(gid) == fid_
                 ? InnerVertexGid2Vertex(gid, v)
                 : OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  DEV_INLINE OID_T GetId(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexId(v) : GetOuterVertexId(v);
  }

  DEV_HOST_INLINE fid_t GetFragId(const vertex_t& u) const {
    if (IsInnerVertex(u)) {
      return fid_;
    } else {
      auto gid = ovgid_[u.GetValue() - ivnum_];
      return id_parser_.get_fragment_id(gid);
    }
  }

  DEV_INLINE const VDATA_T& GetData(const vertex_t& v) const {
    return vdata_[v.GetValue()];
  }

  DEV_INLINE void SetData(const vertex_t& v, const VDATA_T& val) {
    vdata_[v.GetValue()] = val;
  }

  DEV_INLINE VID_T GetLocalOutDegree(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    return oeoffset_[v.GetValue() + 1] - oeoffset_[v.GetValue()];
  }

  DEV_INLINE VID_T GetLocalInDegree(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    return ieoffset_[v.GetValue() + 1] - ieoffset_[v.GetValue()];
  }

  DEV_INLINE bool Gid2Vertex(const VID_T& gid, vertex_t& v) const {
    return id_parser_.get_fragment_id(gid) == fid_
               ? InnerVertexGid2Vertex(gid, v)
               : OuterVertexGid2Vertex(gid, v);
  }

  DEV_INLINE VID_T Vertex2Gid(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

  DEV_HOST_INLINE VID_T GetInnerVerticesNum() const { return ivnum_; }

  DEV_HOST_INLINE VID_T GetOuterVerticesNum() const { return ovnum_; }

  DEV_HOST_INLINE bool IsInnerVertex(const vertex_t& v) const {
    return (v.GetValue() < ivnum_);
  }

  DEV_HOST_INLINE bool IsOuterVertex(const vertex_t& v) const {
    return (v.GetValue() < tvnum_ && v.GetValue() >= ivnum_);
  }

  DEV_INLINE bool GetInnerVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;

    if (vm_.GetGid(fid_, oid, gid)) {
      v.SetValue(id_parser_.get_local_id(gid));
      return true;
    }
    return false;
  }

  DEV_INLINE bool GetOuterVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;
    if (vm_.GetGid(oid, gid)) {
      return OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  DEV_INLINE OID_T GetInnerVertexId(const vertex_t& v) const {
    OID_T internal_oid;
    assert(vm_.GetOid(fid_, v.GetValue(), internal_oid));
    return internal_oid;
  }

  DEV_INLINE OID_T GetOuterVertexId(const vertex_t& v) const {
    VID_T gid = ovgid_[v.GetValue() - ivnum_];

    return Gid2Oid(gid);
  }

  DEV_INLINE OID_T Gid2Oid(const VID_T& gid) const {
    OID_T internal_oid;
    assert(vm_.GetOid(gid, internal_oid));
    return internal_oid;
  }

  DEV_INLINE bool Oid2Gid(const OID_T& oid, VID_T& gid) const {
    return vm_.GetGid(oid, gid);
  }

  DEV_HOST_INLINE bool InnerVertexGid2Vertex(const VID_T& gid,
                                             vertex_t& v) const {
    v.SetValue(id_parser_.get_local_id(gid));
    return true;
  }

  DEV_INLINE bool OuterVertexGid2Vertex(const VID_T& gid, vertex_t& v) const {
    auto iter = ovg2l_->find(gid);
    if (iter != nullptr) {
      v.SetValue(iter->value);
      return true;
    } else {
      return false;
    }
  }

  DEV_INLINE VID_T GetOuterVertexGid(const vertex_t& v) const {
    return ovgid_[v.GetValue() - ivnum_];
  }

  DEV_HOST_INLINE VID_T GetInnerVertexGid(const vertex_t& v) const {
    return id_parser_.generate_global_id(fid_, v.GetValue());
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
  DEV_INLINE DestList IEDests(const vertex_t& v) const {
    assert(!idoffset_.empty());
    assert(IsInnerVertex(v));
    return DestList(idoffset_[v.GetValue()], idoffset_[v.GetValue() + 1]);
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
  DEV_INLINE DestList OEDests(const vertex_t& v) const {
    assert(!odoffset_.empty());
    assert(IsInnerVertex(v));
    return DestList(odoffset_[v.GetValue()], odoffset_[v.GetValue() + 1]);
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
  DEV_INLINE DestList IOEDests(const vertex_t& v) const {
    assert(!iodoffset_.empty());
    assert(IsInnerVertex(v));
    return DestList(iodoffset_[v.GetValue()], iodoffset_[v.GetValue() + 1]);
  }

  DEV_INLINE adj_list_t GetIncomingAdjList(const vertex_t& v) {
    return adj_list_t(ieoffset_[v.GetValue()], ieoffset_[v.GetValue() + 1]);
  }

  DEV_INLINE const_adj_list_t GetIncomingAdjList(const vertex_t& v) const {
    return const_adj_list_t(ieoffset_[v.GetValue()],
                            ieoffset_[v.GetValue() + 1]);
  }

  /**
   * @brief Returns the incoming adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent inner vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return adj_list_t(ieoffset_[v.GetValue()], iespliters_[0][v.GetValue()]);
  }

  /**
   * @brief Returns the incoming adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent inner vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE const_adj_list_t
  GetIncomingInnerVertexAdjList(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return const_adj_list_t(ieoffset_[v.GetValue()],
                            iespliters_[0][v.GetValue()]);
  }
  /**
   * @brief Returns the incoming adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent outer vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return adj_list_t(iespliters_[0][v.GetValue()],
                      ieoffset_[v.GetValue() + 1]);
  }
  /**
   * @brief Returns the incoming adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent outer vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE const_adj_list_t
  GetIncomingOuterVertexAdjList(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return const_adj_list_t(iespliters_[0][v.GetValue()],
                            ieoffset_[v.GetValue() + 1]);
  }
  /**
   * @brief Returns the outgoing adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent inner vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return adj_list_t(oeoffset_[v.GetValue()], oespliters_[0][v.GetValue()]);
  }
  /**
   * @brief Returns the outgoing adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent inner vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE const_adj_list_t
  GetOutgoingInnerVertexAdjList(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return const_adj_list_t(oeoffset_[v.GetValue()],
                            oespliters_[0][v.GetValue()]);
  }

  /**
   * @brief Returns the outgoing adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent outer vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return adj_list_t(oespliters_[0][v.GetValue()],
                      oeoffset_[v.GetValue() + 1]);
  }

  /**
   * @brief Returns the outgoing adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent outer vertices of v.
   *
   * @attention This method is available only when need_split_edges set in
   * application's specification.
   */
  DEV_INLINE const_adj_list_t
  GetOutgoingOuterVertexAdjList(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return const_adj_list_t(oespliters_[0][v.GetValue()],
                            oeoffset_[v.GetValue() + 1]);
  }

  DEV_INLINE adj_list_t GetOutgoingAdjList(const vertex_t& v) {
    return adj_list_t(oeoffset_[v.GetValue()], oeoffset_[v.GetValue() + 1]);
  }

  DEV_INLINE const_adj_list_t GetOutgoingAdjList(const vertex_t& v) const {
    return const_adj_list_t(oeoffset_[v.GetValue()],
                            oeoffset_[v.GetValue() + 1]);
  }

  DEV_INLINE size_t GetIncomingEdgeIndex(const nbr_t& nbr) const {
    size_t edge_idx = &nbr - ie_.data();
    assert(edge_idx < ienum_);
    return edge_idx;
  }

  DEV_INLINE size_t GetIncomingEdgeIndex(const vertex_t& u,
                                         const nbr_t& nbr) const {
    size_t edge_idx = &nbr - ieoffset_[u.GetValue()];
    return edge_idx;
  }

  DEV_INLINE size_t GetOutgoingEdgeIndex(const nbr_t& nbr) const {
    size_t edge_idx = &nbr - oe_.data();
    assert(edge_idx < oenum_);
    return edge_idx;
  }

  DEV_INLINE size_t GetOutgoingEdgeIndex(const vertex_t& u,
                                         const nbr_t& nbr) const {
    size_t edge_idx = &nbr - oeoffset_[u.GetValue()];
    return edge_idx;
  }

 private:
  DeviceVertexMap<OID_T, VID_T> vm_;
  size_t ivnum_{}, ovnum_{}, tvnum_{};
  size_t ienum_{}, oenum_{};

  fid_t fid_{};

  IdParser<VID_T> id_parser_;

  CUDASTL::HashMap<VID_T, VID_T>* ovg2l_{};
  ArrayView<VID_T> ovgid_{};

  ArrayView<nbr_t*> ieoffset_, oeoffset_;
  ArrayView<nbr_t> ie_, oe_;
  ArrayView<VDATA_T> vdata_{};

  ArrayView<fid_t> idst_, odst_, iodst_;
  ArrayView<fid_t*> idoffset_, odoffset_, iodoffset_;

  ArrayView<ArrayView<nbr_t*>> iespliters_, oespliters_;

  ArrayView<vertex_range_t> outer_vertices_of_frag_;

  ArrayView<ArrayView<vertex_t>> mirrors_of_frag_;

  template <typename _OID_T, typename _VID_T, typename _VDATA_T,
            typename _EDATA_T, grape::LoadStrategy __load_strategy,
            typename _VERTEX_MAP_T>
  friend class grape::cuda::HostFragment;
};

}  // namespace dev
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_FRAGMENT_DEVICE_FRAGMENT_H_
