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

#ifndef GRAPE_FRAGMENT_EDGECUT_FRAGMENT_BASE_H_
#define GRAPE_FRAGMENT_EDGECUT_FRAGMENT_BASE_H_

#include "grape/fragment/fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/utils/vertex_array.h"

namespace grape {

/**
 * @brief IEdgecutFragment defines the interfaces of fragments with edgecut.
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
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class EdgecutFragmentBase
    : virtual public FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T> {
 public:
  using vid_t = VID_T;
  using vertex_t = Vertex<VID_T>;

  /**
   * @brief Returns the number of inner vertices in this fragment.
   *
   * @return The number of inner vertices in this fragment.
   */
  virtual VID_T GetInnerVerticesNum() const = 0;

  /**
   * @brief Returns the number of outer vertices in this fragment.
   *
   * @return The number of outer vertices in this fragment.
   */
  virtual VID_T GetOuterVerticesNum() const = 0;

  /**
   * @brief Returns the vertex range of inner vertices in this fragment.
   *
   * @return The vertex range of inner vertices in this fragment.
   */
  virtual VertexRange<VID_T> InnerVertices() const = 0;

  /**
   * @brief Returns the vertex range of outer vertices in this fragment.
   *
   * @return The vertex range of outer vertices in this fragment.
   */
  virtual VertexRange<VID_T> OuterVertices() const = 0;

  /**
   * @brief Check if vertex v is inner vertex of this fragment.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is an inner vertex, false otherwise.
   */
  virtual bool IsInnerVertex(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Check if vertex v is outer vertex of this fragment.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is outer vertex, false otherwise.
   */
  virtual bool IsOuterVertex(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Get a inner vertex with original ID vid.
   *
   * @param vid Original ID.
   * @param v Got vertex.
   *
   * @return True if find a inner vertex with original ID vid in this fragment,
   * false otherwise.
   */
  virtual bool GetInnerVertex(const OID_T& vid, Vertex<VID_T>& v) const = 0;

  /**
   * @brief Get a outer vertex with original ID vid.
   *
   * @param vid Original ID.
   * @param v Got vertex.
   *
   * @return True if find a outer vertex with original ID vid in this fragment,
   * false otherwise.
   */
  virtual bool GetOuterVertex(const OID_T& vid, Vertex<VID_T>& v) const = 0;

  /**
   * @brief Get the original ID of an inner vertex.
   *
   * @param v Input vertex.
   *
   * @return The original ID.
   */
  virtual OID_T GetInnerVertexId(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Get the original ID of an outer vertex.
   *
   * @param v Input vertex.
   *
   * @return The original ID.
   */
  virtual OID_T GetOuterVertexId(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Convert from global id to an inner vertex handle.
   *
   * @param gid Input global id.
   * @param v Output vertex handle.
   *
   * @return True if exists an inner vertex of this fragment with global id as
   * gid, false otherwise.
   */
  virtual bool InnerVertexGid2Vertex(const VID_T& gid,
                                     Vertex<VID_T>& v) const = 0;

  /**
   * @brief Convert from global id to an outer vertex handle.
   *
   * @param gid Input global id.
   * @param v Output vertex handle.
   *
   * @return True if exists an outer vertex of this fragment with global id as
   * gid, false otherwise.
   */
  virtual bool OuterVertexGid2Vertex(const VID_T& gid,
                                     Vertex<VID_T>& v) const = 0;

  /**
   * @brief Convert from inner vertex handle to its global id.
   *
   * @param v Input vertex handle.
   *
   * @return Global id of the vertex.
   */
  virtual VID_T GetOuterVertexGid(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Convert from outer vertex handle to its global id.
   *
   * @param v Input vertex handle.
   *
   * @return Global id of the vertex.
   */
  virtual VID_T GetInnerVertexGid(const Vertex<VID_T>& v) const = 0;

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
  virtual AdjList<VID_T, EDATA_T> GetIncomingInnerVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the incoming adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent inner vertices of v.
   */
  virtual ConstAdjList<VID_T, EDATA_T> GetIncomingInnerVertexAdjList(
      const Vertex<VID_T>& v) const = 0;
  /**
   * @brief Returns the incoming adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent outer vertices of v.
   */
  virtual AdjList<VID_T, EDATA_T> GetIncomingOuterVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the incoming adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent outer vertices of v.
   */
  virtual ConstAdjList<VID_T, EDATA_T> GetIncomingOuterVertexAdjList(
      const Vertex<VID_T>& v) const = 0;
  /**
   * @brief Returns the outgoing adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent inner vertices of v.
   */
  virtual AdjList<VID_T, EDATA_T> GetOutgoingInnerVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the outgoing adjacent inner vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent inner vertices of v.
   */
  virtual ConstAdjList<VID_T, EDATA_T> GetOutgoingInnerVertexAdjList(
      const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Returns the outgoing adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent outer vertices of v.
   */
  virtual AdjList<VID_T, EDATA_T> GetOutgoingOuterVertexAdjList(
      const Vertex<VID_T>& v) = 0;
  /**
   * @brief Returns the outgoing adjacent outer vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent outer vertices of v.
   */
  virtual ConstAdjList<VID_T, EDATA_T> GetOutgoingOuterVertexAdjList(
      const Vertex<VID_T>& v) const = 0;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_EDGECUT_FRAGMENT_BASE_H_
