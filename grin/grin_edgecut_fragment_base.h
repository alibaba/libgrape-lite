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

#ifndef GRIN_GRIN_EDGECUT_FRAGMENT_BASE_H_
#define GRIN_GRIN_EDGECUT_FRAGMENT_BASE_H_

#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/utils/vertex_array.h"

#include "grin/grin_fragment_base.h"

extern "C" {
#include "grin/include/partition/partition.h"
#include "grin/include/topology/vertexlist.h"
}

namespace grape {

/**
 * @brief GRIN_EdgecutFragment is the GRIN version of IEdgecutFragment in grape,
 * which defines the interfaces of fragments with edgecut.
 * To learn more about edge-cut and vertex-cut, please refers to
 * https://spark.apache.org/docs/1.6.2/graphx-programming-guide.html#optimized-representation
 *
 * If we have an edge a->b cutted by the partitioner, and a is in frag_0, and b
 * in frag_1. Then:
 * a->b is a crossing edge,
 * a is an inner_vertex in frag_0,
 * b is an outer_vertex in frag_0.
 *
 * @tparam OID_T
 * @tparam VID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 * @tparam VERTEX_MAP_T
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T>
class GRIN_EdgecutFragmentBase
    : virtual public GRIN_FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T,
                                       VERTEX_MAP_T> {
 public:
  using base_t = GRIN_FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, VERTEX_MAP_T>;
  using vid_t = VID_T;
  using vertex_t = Vertex<VID_T>;
  using base_t::fid_;
  using base_t::pg_;

  GRIN_EdgecutFragmentBase() {}

  /**
   * @brief Returns the number of inner vertices in this fragment.
   *
   * @return The number of inner vertices in this fragment.
   */
  VID_T GetInnerVerticesNum() const {
    auto vl = get_local_vertices(pg_, fid_);
    return get_vertex_list_size(vl);
  }

  /**
   * @brief Returns the number of outer vertices in this fragment.
   *
   * @return The number of outer vertices in this fragment.
   */
  VID_T GetOuterVerticesNum() const {
    auto vl = get_remote_vertices(pg_, fid_);
    return get_vertex_list_size(vl);
  }

  using inner_vertices_t = VertexRange<VID_T>;
  /**
   * @brief Returns the vertex range of inner vertices in this fragment.
   *
   * @return The vertex range of inner vertices in this fragment.
   */
  const inner_vertices_t InnerVertices() const {
    void* vlh = get_local_vertices(pg_, fid_);
    void* beginh = get_begin_vertex_id_from_list(vlh);
    VID_T* begin = static_cast<VID_T*>(beginh);
    void* endh = get_end_vertex_id_from_list(vlh);
    VID_T* end = static_cast<VID_T*>(endh);
    return inner_vertices_t(*begin, *end);
  }

  using outer_vertices_t = VertexRange<VID_T>;
  /**
   * @brief Returns the vertex range of outer vertices in this fragment.
   *
   * @return The vertex range of outer vertices in this fragment.
   */
  const outer_vertices_t OuterVertices() const {
    void* vlh = get_remote_vertices(pg_, fid_);
    void* beginh = get_begin_vertex_id_from_list(vlh);
    VID_T* begin = static_cast<VID_T*>(beginh);
    void* endh = get_end_vertex_id_from_list(vlh);
    VID_T* end = static_cast<VID_T*>(endh);
    return outer_vertices_t(*begin, *end);
  }

  using sub_vertices_t = VertexRange<VID_T>;
  const sub_vertices_t OuterVertices(fid_t fid) const {
    void* vlh = get_remote_vertices_by_partition(pg_, fid_);
    void* beginh = get_begin_vertex_id_from_list(vlh);
    VID_T* begin = static_cast<VID_T*>(beginh);
    void* endh = get_end_vertex_id_from_list(vlh);
    VID_T* end = static_cast<VID_T*>(endh);
    return sub_vertices_t(*begin, *end);
  }

  using mirror_vertices_t = std::vector<Vertex<VID_T>>;
  const mirror_vertices_t MirrorVertices(fid_t fid) const {
    return mirrors_of_frag_[fid];
  }

  /**
   * @brief Check if vertex v is inner vertex of this fragment.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is an inner vertex, false otherwise.
   */
  bool IsInnerVertex(const Vertex<VID_T>& v) const {
    return is_local_vertex(pg_, fid_, (void*)(&v));
  }

  /**
   * @brief Check if vertex v is outer vertex of this fragment.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is outer vertex, false otherwise.
   */
  bool IsOuterVertex(const Vertex<VID_T>& v) const {
    return !is_local_vertex(pg_, fid_, (void*)(&v));
  }

  /**
   * @brief Get a inner vertex with original ID vid.
   *
   * @param vid Original ID.
   * @param v Got vertex.
   *
   * @return True if find a inner vertex with original ID vid in this fragment,
   * false otherwise.
   */
  bool GetInnerVertex(const OID_T& oid, Vertex<VID_T>& v) const {
    VID_T gid;
    if (vm_ptr_->GetGid(fid(), oid, gid)) {
      return InnerVertexGid2Vertex(gid, v);
    }
    return false;
  }

  /**
   * @brief Get the original ID of an inner vertex.
   *
   * @param v Input vertex.
   *
   * @return The original ID.
   */
  OID_T GetInnerVertexId(vertex_t v) const {
    OID_T oid;
    vm_ptr_->GetOid(GetInnerVertexGid(v), oid);
    return oid;
  }

  /**
   * @brief Get the original ID of an outer vertex.
   *
   * @param v Input vertex.
   *
   * @return The original ID.
   */
  OID_T GetOuterVertexId(vertex_t v) const {
    OID_T oid;
    vm_ptr_->GetOid(GetOuterVertexGid(v), oid);
    return oid;
  }

  /**
   * @brief Convert from global id to an inner vertex handle.
   *
   * @param gid Input global id.
   * @param v Output vertex handle.
   *
   * @return True if exists an inner vertex of this fragment with global id as
   * gid, false otherwise.
   */
  inline bool InnerVertexGid2Vertex(VID_T gid, Vertex<VID_T>& v) const {
    return Gid2Vertex(gid, v);
  }

  /**
   * @brief Convert from global id to an outer vertex handle.
   *
   * @param gid Input global id.
   * @param v Output vertex handle.
   *
   * @return True if exists an outer vertex of this fragment with global id as
   * gid, false otherwise.
   */
  inline bool OuterVertexGid2Vertex(VID_T gid, Vertex<VID_T>& v) const {
    return Gid2Vertex(gid, v);
  }

  /**
   * @brief Convert from inner vertex handle to its global id.
   *
   * @param v Input vertex handle.
   *
   * @return Global id of the vertex.
   */
  virtual VID_T GetOuterVertexGid(vertex_t v) const = 0;

  /**
   * @brief Convert from outer vertex handle to its global id.
   *
   * @param v Input vertex handle.
   *
   * @return Global id of the vertex.
   */
  VID_T GetInnerVertexGid(vertex_t v) const {
    std::stringstream ss(serialize_remote_vertex(pg_, (void*)(&v)));
    VID_T gid;
    ss >> gid;
    return gid;
  }

  /**
   * @brief Return the incoming edge destination fragment ID list of a inner
   * vertex.
   *
   * @note: For inner vertex v of fragment-0, if outer vertex u and w are
   * parents of v. u belongs to fragment-1 and w belongs to fragment-2, then 1
   * and 2 are in incoming edge destination fragment ID list of v.
   * @note: This method is encapsulated in the corresponding sending message
   * API, SendMsgThroughIEdges, so it is not recommended to use this method
   * directly in application programs.
   *
   * @param v Input vertex.
   *
   * @return The incoming edge destination fragment ID list.
   */
  virtual DestList IEDests(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Return the outgoing edge destination fragment ID list of a inner
   * vertex.
   *
   * @note: For inner vertex v of fragment-0, if outer vertex u and w are
   * children of v. u belongs to fragment-1 and w belongs to fragment-2, then 1
   * and 2 are in outgoing edge destination fragment ID list of v.
   * @note: This method is encapsulated in the corresponding sending message
   * API, SendMsgThroughOEdges, so it is not recommended to use this method
   * directly in application programs.
   *
   * @param v Input vertex.
   *
   * @return The outgoing edge destination fragment ID list.
   */
  virtual DestList OEDests(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Return the edge destination fragment ID list of a inner vertex.
   *
   * @note: For inner vertex v of fragment-0, if outer vertex u and w are
   * neighbors of v. u belongs to fragment-1 and w belongs to fragment-2, then 1
   * and 2 are in outgoing edge destination fragment ID list of v.
   * @note: This method is encapsulated in the corresponding sending message
   * API, SendMsgThroughEdges, so it is not recommended to use this method
   * directly in application programs.
   *
   * @param v Input vertex.
   *
   * @return The edge destination fragment ID list.
   */
  virtual DestList IOEDests(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Returns the incoming adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent inner vertices of v.
   */
  virtual GRIN_AdjList<VID_T, EDATA_T> GetIncomingInnerVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the incoming adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent inner vertices of v.
   */
  virtual GRIN_ConstAdjList<VID_T, EDATA_T> GetIncomingInnerVertexAdjList(
      const Vertex<VID_T>& v) const = 0;
  /**
   * @brief Returns the incoming adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent outer vertices of v.
   */
  virtual GRIN_AdjList<VID_T, EDATA_T> GetIncomingOuterVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the incoming adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent outer vertices of v.
   */
  virtual GRIN_ConstAdjList<VID_T, EDATA_T> GetIncomingOuterVertexAdjList(
      const Vertex<VID_T>& v) const = 0;
  /**
   * @brief Returns the outgoing adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent inner vertices of v.
   */
  virtual GRIN_AdjList<VID_T, EDATA_T> GetOutgoingInnerVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the outgoing adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent inner vertices of v.
   */
  virtual GRIN_ConstAdjList<VID_T, EDATA_T> GetOutgoingInnerVertexAdjList(
      const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Returns the outgoing adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent outer vertices of v.
   */
  virtual GRIN_AdjList<VID_T, EDATA_T> GetOutgoingOuterVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the outgoing adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent outer vertices of v.
   */
  virtual GRIN_ConstAdjList<VID_T, EDATA_T> GetOutgoingOuterVertexAdjList(
      const Vertex<VID_T>& v) const = 0;

  using base_t::fid;
  using base_t::fnum;

  bool Gid2Vertex(const vid_t& gid, vertex_t& v) const override {
    std::stringstream ss;
    ss << gid;
    void* vh = get_vertex_from_deserialization(pg_, fid_, ss.str().c_str());
    if (vh == NULL_VERTEX) {
      return false;
    }
    vertex_t* _v = static_cast<vertex_t*>(vh);
    v.SetValue(_v->GetValue());
    return true;
  }

  vid_t Vertex2Gid(const vertex_t& v) const override {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

 protected:
  virtual bool OuterVertexGid2Lid(VID_T gid, VID_T& lid) const = 0;

  void initMirrorInfo(const CommSpec& comm_spec) {
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    mirrors_of_frag_.resize(fnum());

    std::thread send_thread([&]() {
      std::vector<vertex_t> gid_list;
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker_id = (worker_id + i) % worker_num;
        fid_t dst_fid = comm_spec.WorkerToFrag(dst_worker_id);
        auto& range = OuterVertices(dst_fid);
        gid_list.clear();
        gid_list.reserve(range.size());
        for (auto& v : range) {
          gid_list.emplace_back(
              get_master_partition_for_vertex(pg_, fid_, (void*)(&v)));
        }
        sync_comm::Send<std::vector<vertex_t>>(gid_list, dst_worker_id, 0,
                                               comm_spec.comm());
      }
    });

    std::thread recv_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int src_worker_id = (worker_id + worker_num - i) % worker_num;
        fid_t src_fid = comm_spec.WorkerToFrag(src_worker_id);
        auto& mirror_vec = mirrors_of_frag_[src_fid];
        sync_comm::Recv<std::vector<vertex_t>>(mirror_vec, src_worker_id, 0,
                                               comm_spec.comm());
      }
    });

    recv_thread.join();
    send_thread.join();
  }

  std::vector<mirror_vertices_t> mirrors_of_frag_;
  using base_t::vm_ptr_;
};

}  // namespace grape

#endif  // GRIN_GRIN_EDGECUT_FRAGMENT_BASE_H_
