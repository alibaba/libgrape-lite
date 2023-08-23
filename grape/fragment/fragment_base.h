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

#ifndef GRAPE_FRAGMENT_FRAGMENT_BASE_H_
#define GRAPE_FRAGMENT_FRAGMENT_BASE_H_

#include <vector>

#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"

namespace grape {
/**
 * @brief FragmentBase is the base class for fragments.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the
 * derived classes would be invoked directly, not via virtual functions.
 *
 * @tparam OID_T
 * @tparam VID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class FragmentBase {
 public:
  /**
   * @brief Construct a fragment with a set of vertices and edges.
   *
   * @param fid Fragment ID
   * @param vertices A set of vertices.
   * @param edges A set of edges.
   */
  virtual void Init(fid_t fid,
                    std::vector<internal::Vertex<VID_T, VDATA_T>>& vertices,
                    std::vector<Edge<VID_T, EDATA_T>>& edges) = 0;

  /**
   * @brief For some kind of applications, specific data structures will be
   * generated.
   *
   * @param strategy
   * @param need_split_edge
   */
  virtual void PrepareToRunApp(MessageStrategy strategy,
                               bool need_split_edge) = 0;

  /**
   * @brief Returns the ID of this fragment.
   *
   * @return The ID of this fragment.
   */
  virtual fid_t fid() const = 0;

  /**
   * @brief Returns the number of fragments.
   *
   * @return The number of fragments.
   */
  virtual fid_t fnum() const = 0;

  /**
   * @brief Returns the number of edges in this fragment.
   *
   * @return The number of edges in this fragment.
   */
  virtual size_t GetEdgeNum() const = 0;

  /**
   * @brief Returns the number of vertices in this fragment.
   *
   * @return The number of vertices in this fragment.
   */
  virtual VID_T GetVerticesNum() const = 0;

  /**
   * @brief Returns the number of vertices in the entire graph.
   *
   * @return The number of vertices in the entire graph.
   */
  virtual size_t GetTotalVerticesNum() const = 0;

  /**
   * @brief Get all vertices referenced to this fragment.
   *
   * @return A vertex set can be iterate on.
   */
  virtual VertexRange<VID_T> Vertices() const = 0;

  /**
   * @brief Get a vertex with original ID vid.
   *
   * @param vid Original ID.
   * @param v Got vertex.
   *
   * @return If find the vertex in this fragment, return true. Otherwise, return
   * false.
   */
  virtual bool GetVertex(const OID_T& vid, Vertex<VID_T>& v) const = 0;

  /**
   * @brief Get the original ID of a vertex.
   *
   * @param v Input vertex.
   *
   * @return Its original ID.
   */
  virtual OID_T GetId(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Get the ID of fragment the input vertex belongs to.
   *
   * @param u Input vertex.
   *
   * @return Its fragment ID.
   */
  virtual fid_t GetFragId(const Vertex<VID_T>& u) const = 0;

  /**
   * @brief Get the data of a vertex.
   *
   * @param v Input vertex.
   *
   * @return Data on it.
   */
  virtual const VDATA_T& GetData(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Set the data of a vertex.
   *
   * @param v Input vertex.
   * @param val Data to write.
   * @attention This will only be applied locally, won't sync on other mirrors
   * globally.
   */
  virtual void SetData(const Vertex<VID_T>& v, const VDATA_T& val) = 0;

  /**
   * @brief Check if vertex v has a child, that is, existing an edge v->u.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v has a child, false otherwise.
   */
  virtual bool HasChild(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Check if vertex v has a parent, that is, existing an edge u->v.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v has a parent, false otherwise.
   */
  virtual bool HasParent(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Returns the in-degree of vertex v in this fragment.
   *
   * @param v Input vertex.
   *
   * @return In-degree of vertex v in this fragment.
   */
  virtual int GetLocalInDegree(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Returns the out-degree of vertex v in this fragment.<Paste>
   *
   * @param v Input vertex.
   *
   * @return Out-degree of vertex v in this fragment.
   */
  virtual int GetLocalOutDegree(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Convert from global id to a vertex handle.
   *
   * @param gid Input global id.
   * @param v Output vertex handle.
   *
   * @return True if exists a vertex with global id as gid in this fragment,
   * false otherwise.
   */
  virtual bool Gid2Vertex(const VID_T& gid, Vertex<VID_T>& v) const = 0;

  /**
   * @brief Convert from vertex handle to its global id.
   *
   * @param v Input vertex handle.
   *
   * @return Global id of the vertex.
   */
  virtual VID_T Vertex2Gid(const Vertex<VID_T>& v) const = 0;

  /**
   * @brief Returns the incoming adjacent vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The incoming adjacent vertices of v.
   */
  virtual AdjList<VID_T, EDATA_T> GetIncomingAdjList(
      const Vertex<VID_T>& v) = 0;

  virtual ConstAdjList<VID_T, EDATA_T> GetIncomingAdjList(
      const Vertex<VID_T>& v) const = 0;
  /**
   * @brief Returns the outgoing adjacent vertices of v.
   *
   * @param v Input vertex.
   *
   * @return The outgoing adjacent vertices of v.
   */
  virtual AdjList<VID_T, EDATA_T> GetOutgoingAdjList(
      const Vertex<VID_T>& v) = 0;

  virtual ConstAdjList<VID_T, EDATA_T> GetOutgoingAdjList(
      const Vertex<VID_T>& v) const = 0;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_FRAGMENT_BASE_H_
