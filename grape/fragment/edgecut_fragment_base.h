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
 * @tparam TRAITS_T
 */
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename TRAITS_T>
class EdgecutFragmentBase
    : virtual public FragmentBase<OID_T, VDATA_T, EDATA_T> {
 public:
  using base_t = FragmentBase<OID_T, VDATA_T, EDATA_T>;
  using vid_t = VID_T;
  using vertex_t = Vertex<VID_T>;
  using fragment_adj_list_t = typename TRAITS_T::fragment_adj_list_t;
  using fragment_const_adj_list_t =
      typename TRAITS_T::fragment_const_adj_list_t;

  EdgecutFragmentBase() {}

  static constexpr FragmentType fragment_type = FragmentType::kEdgeCut;

  VertexMap<OID_T, VID_T>& GetVertexMap() { return *vm_ptr_; }
  const VertexMap<OID_T, VID_T>& GetVertexMap() const { return *vm_ptr_; }

  using vertices_t = typename TRAITS_T::vertices_t;
  /**
   * @brief Get all vertices referenced to this fragment.
   *
   * @return A vertex set can be iterate on.
   */
  const vertices_t& Vertices() const { return vertices_; }

  size_t GetVerticesNum() const override { return vertices_.size(); }

  size_t GetTotalVerticesNum() const override {
    return vm_ptr_->GetTotalVertexSize();
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
      return Gid2Vertex(gid, v);
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
    OID_T oid{};
    vm_ptr_->GetOid(Vertex2Gid(v), oid);
    return oid;
  }

  using internal_id_t = typename InternalOID<OID_T>::type;

  internal_id_t GetInternalId(const Vertex<VID_T>& v) const {
    internal_id_t oid{};
    vm_ptr_->GetInternalOid(Vertex2Gid(v), oid);
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
    VID_T gid = Vertex2Gid(u);
    return id_parser_.get_fragment_id(gid);
  }

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

  /**
   * @brief Returns the number of inner vertices in this fragment.
   *
   * @return The number of inner vertices in this fragment.
   */
  VID_T GetInnerVerticesNum() const { return inner_vertices_.size(); }

  /**
   * @brief Returns the number of outer vertices in this fragment.
   *
   * @return The number of outer vertices in this fragment.
   */
  VID_T GetOuterVerticesNum() const { return outer_vertices_.size(); }

  using inner_vertices_t = typename TRAITS_T::inner_vertices_t;
  /**
   * @brief Returns the vertex range of inner vertices in this fragment.
   *
   * @return The vertex range of inner vertices in this fragment.
   */
  const inner_vertices_t& InnerVertices() const { return inner_vertices_; }

  using outer_vertices_t = typename TRAITS_T::outer_vertices_t;
  /**
   * @brief Returns the vertex range of outer vertices in this fragment.
   *
   * @return The vertex range of outer vertices in this fragment.
   */
  const outer_vertices_t& OuterVertices() const { return outer_vertices_; }

  using sub_vertices_t = typename TRAITS_T::sub_vertices_t;
  const sub_vertices_t& OuterVertices(fid_t fid) const {
    return outer_vertices_of_frag_[fid];
  }

  using mirror_vertices_t = typename TRAITS_T::mirror_vertices_t;
  const mirror_vertices_t& MirrorVertices(fid_t fid) const {
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
    return inner_vertices_.Contain(v);
  }

  /**
   * @brief Check if vertex v is outer vertex of this fragment.
   *
   * @param v Input vertex.
   *
   * @return True if vertex v is outer vertex, false otherwise.
   */
  bool IsOuterVertex(const Vertex<VID_T>& v) const {
    return outer_vertices_.Contain(v);
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
    OID_T oid{};
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
    OID_T oid{};
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
    VID_T lid = id_parser_.get_local_id(gid);
    v.SetValue(lid);
    return true;
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
    VID_T lid;
    if (OuterVertexGid2Lid(gid, lid)) {
      v.SetValue(lid);
      return true;
    }
    return false;
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
    return id_parser_.generate_global_id(fid(), v.GetValue());
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

  virtual size_t IEDestsSize() const = 0;

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

  virtual size_t OEDestsSize() const = 0;

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

  virtual size_t IOEDestsSize() const = 0;

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

  using base_t::fid;
  using base_t::fnum;

  /**
   * @brief Convert from global id to a vertex handle.
   *
   * @param gid Input global id.
   * @param v Output vertex handle.
   *
   * @return True if exists a vertex with global id as gid in this fragment,
   * false otherwise.
   */
  bool Gid2Vertex(const vid_t& gid, vertex_t& v) const {
    return IsInnerVertexGid(gid) ? InnerVertexGid2Vertex(gid, v)
                                 : OuterVertexGid2Vertex(gid, v);
  }

  /**
   * @brief Convert from vertex handle to its global id.
   *
   * @param v Input vertex handle.
   *
   * @return Global id of the vertex.
   */
  vid_t Vertex2Gid(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

 protected:
  void init(fid_t fid, bool directed,
            std::unique_ptr<VertexMap<OID_T, VID_T>>&& vm_ptr) {
    fid_t fnum = vm_ptr->GetFragmentNum();
    base_t::init(fid, fnum, directed);
    vm_ptr_ = std::move(vm_ptr);
    ivnum_ = vm_ptr_->GetInnerVertexSize(fid);
    id_parser_.init(fnum);
  }

  inline bool IsInnerVertexGid(VID_T gid) const {
    return id_parser_.get_fragment_id(gid) == fid();
  }

  inline bool IsInnerVertexLid(VID_T lid) const {
    return inner_vertices_.Contain(vertex_t(lid));
  }

  inline bool Gid2Lid(VID_T gid, VID_T& lid) const {
    return IsInnerVertexGid(gid) ? InnerVertexGid2Lid(gid, lid)
                                 : OuterVertexGid2Lid(gid, lid);
  }

  inline bool InnerVertexGid2Lid(VID_T gid, VID_T& lid) const {
    lid = id_parser_.get_local_id(gid);
    return true;
  }

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
          gid_list.emplace_back(id_parser_.get_local_id(Vertex2Gid(v)));
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

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    base_t::serialize(writer);
    InArchive arc;
    arc << ivnum_ << inner_vertices_ << outer_vertices_ << vertices_;
    CHECK(writer->WriteArchive(arc));
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    base_t::deserialize(reader);
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> ivnum_ >> inner_vertices_ >> outer_vertices_ >> vertices_;
    id_parser_.init(fnum());
  }

  VID_T ivnum_;

  inner_vertices_t inner_vertices_;
  outer_vertices_t outer_vertices_;
  vertices_t vertices_;

  std::vector<sub_vertices_t> outer_vertices_of_frag_;
  std::vector<mirror_vertices_t> mirrors_of_frag_;

  std::unique_ptr<VertexMap<OID_T, VID_T>> vm_ptr_;
  IdParser<VID_T> id_parser_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_EDGECUT_FRAGMENT_BASE_H_
