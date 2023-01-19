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

#ifndef GRIN_GRIN_FRAGMENT_BASE_H_
#define GRIN_GRIN_FRAGMENT_BASE_H_

#include <vector>

#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/worker/comm_spec.h"
#include "grin/include/predefine.h"

extern "C" {
#include "grin/include/partition/partition.h"
#include "grin/include/topology/structure.h"
#include "grin/include/topology/vertexlist.h"
}

namespace grape {

/**
 * @brief GRIN_FragmentBase is the GRIN version of FragmentBase in grape,
 * which is the base class for fragments.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the
 * derived classes would be invoked directly, not via virtual functions.
 *
 * @tparam OID_T
 * @tparam VID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 * @tparam TRAITS_T
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename TRAITS_T>
class GRIN_FragmentBase {
 public:
  using vertex_map_t = typename TRAITS_T::vertex_map_t;
  using fragment_adj_list_t = typename TRAITS_T::fragment_adj_list_t;
  using fragment_const_adj_list_t =
      typename TRAITS_T::fragment_const_adj_list_t;

  GRIN_FragmentBase() : pg_(nullptr) {}

  explicit GRIN_FragmentBase(std::shared_ptr<vertex_map_t> vm_ptr)
      : vm_ptr_(vm_ptr) {}

  std::shared_ptr<vertex_map_t> GetVertexMap() { return vm_ptr_; }
  const std::shared_ptr<vertex_map_t> GetVertexMap() const { return vm_ptr_; }
  void SetVertexMap(std::shared_ptr<vertex_map_t> vm_ptr) { vm_ptr_ = vm_ptr; }

 protected:
  void init(void* partitioned_graph) {
    pg_ = partitioned_graph;
    assert(get_partition_list_size(pg_) == 1);
    auto pl = get_local_partitions(pg_);
    auto p = get_partition_from_list(pl, 0);
    g_ = get_local_graph_from_partition(pg_, p);

    directed_ = is_directed(g_);
    fid_ = p;
    fnum_ = get_total_partitions_number(pg_);
  }

 public:
  /**
   * @brief For some kind of applications, specific data structures will be
   * generated.
   *
   * @param strategy
   * @param need_split_edge
   */
  virtual void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) = 0;

  /**
   * @brief Returns true if the fragment is directed, false otherwise.
   *
   * @return true if the fragment is directed, false otherwise.
   */
  bool directed() const { return directed_; }

  /**
   * @brief Returns the ID of this fragment.
   *
   * @return The ID of this fragment.
   */
  fid_t fid() const { return fid_; }

  /**
   * @brief Returns the number of fragments.
   *
   * @return The number of fragments.
   */
  fid_t fnum() const { return fnum_; }

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
  VID_T GetVerticesNum() const {
    auto vl = get_vertex_list(g_);
    return get_vertex_list_size(vl);
  }

  /**
   * @brief Returns the number of vertices in the entire graph.
   *
   * @return The number of vertices in the entire graph.
   */
  size_t GetTotalVerticesNum() const { return get_total_vertices_number(pg_); }

  using vertices_t = typename TRAITS_T::vertices_t;
  /**
   * @brief Get all vertices referenced to this fragment.
   *
   * @return A vertex set can be iterate on.
   */
  const vertices_t Vertices() const {
    auto vl = get_vertex_list(g_);
    auto viter = get_vertex_list_begin(vl);
    return vertices_t(viter, viter + get_vertex_list_size(vl));
  }

  /**
   * @brief Get a vertex with original ID vid.
   *
   * @param vid Original ID.
   * @param v Got vertex.
   *
   * @return If find the vertex in this fragment, return true. Otherwise, return
   * false.
   */
  bool GetVertex(const OID_T& oid, Vertex<VID_T>& v) const {
    VID_T gid;
    if (vm_ptr_->GetGid(oid, gid)) {
      std::stringstream ss;
      ss << gid;
      auto vh = get_vertex_from_deserialization(pg_, fid_, ss.str().c_str());
      if (vh != NULL_VERTEX) {
        v.SetValue(vh);
        return true;
      } else {
        return false;
      }
    }
    return false;
  }

  /**
   * @brief Get the original ID of a vertex.
   *
   * @param v Input vertex.
   *
   * @return Its original ID.
   */
  OID_T GetId(const Vertex<VID_T>& v) const {
    OID_T oid;
    vm_ptr_->GetOid(Vertex2Gid(v), oid);
    return oid;
  }

  OID_T Gid2Oid(VID_T gid) const {
    OID_T oid;
    vm_ptr_->GetOid(gid, oid);
    return oid;
  }

  /**
   * @brief Get the ID of fragment the input vertex belongs to.
   *
   * @param u Input vertex.
   *
   * @return Its fragment ID.
   */
  fid_t GetFragId(const Vertex<VID_T>& u) const {
    auto rp = get_master_partition_for_vertex(pg_, fid_, u.GetValue());
    if (rp == NULL_REMOTE_PARTITION) {
      return fid_;
    }
    return rp;
  }

  /**
   * @brief Get the data of a vertex.
   *
   * @param v Input vertex.
   *
   * @return Data on it.
   */
  virtual const VDATA_T GetData(const Vertex<VID_T>& v) const = 0;

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

  virtual fragment_adj_list_t GetIncomingAdjList(const Vertex<VID_T>& v,
                                                 fid_t fid) = 0;

  virtual fragment_const_adj_list_t GetIncomingAdjList(const Vertex<VID_T>& v,
                                                       fid_t fid) const = 0;
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

  virtual fragment_adj_list_t GetOutgoingAdjList(const Vertex<VID_T>& v,
                                                 fid_t fid) = 0;

  virtual fragment_const_adj_list_t GetOutgoingAdjList(const Vertex<VID_T>& v,
                                                       fid_t fid) const = 0;

 protected:
  void* pg_;
  void* g_;
  fid_t fid_, fnum_;
  bool directed_;

  std::shared_ptr<vertex_map_t> vm_ptr_;
};

}  // namespace grape

#endif  // GRIN_GRIN_FRAGMENT_BASE_H_
