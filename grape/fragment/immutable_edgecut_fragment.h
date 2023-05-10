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

#ifndef GRAPE_FRAGMENT_IMMUTABLE_EDGECUT_FRAGMENT_H_
#define GRAPE_FRAGMENT_IMMUTABLE_EDGECUT_FRAGMENT_H_

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

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/config.h"
#include "grape/fragment/csr_edgecut_fragment_base.h"
#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/immutable_csr.h"
#include "grape/graph/vertex.h"
#include "grape/io/io_adaptor_base.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/types.h"
#include "grape/util.h"
#include "grape/utils/vertex_array.h"
#include "grape/vertex_map/global_vertex_map.h"
#include "grape/worker/comm_spec.h"

namespace grape {
class CommSpec;
class OutArchive;

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T>
struct ImmutableEdgecutFragmentTraits {
  using inner_vertices_t = VertexRange<VID_T>;
  using outer_vertices_t = VertexRange<VID_T>;
  using vertices_t = VertexRange<VID_T>;
  using sub_vertices_t = VertexRange<VID_T>;

  using fragment_adj_list_t = AdjList<VID_T, EDATA_T>;
  using fragment_const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;

  using csr_t = ImmutableCSR<VID_T, Nbr<VID_T, EDATA_T>>;
  using csr_builder_t = ImmutableCSRBuild<VID_T, Nbr<VID_T, EDATA_T>>;
  using mirror_vertices_t = std::vector<Vertex<VID_T>>;
  using vertex_map_t = VERTEX_MAP_T;
};

/**
 * @brief A kind of edgecut fragment.
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
class ImmutableEdgecutFragment
    : public CSREdgecutFragmentBase<
          OID_T, VID_T, VDATA_T, EDATA_T,
          ImmutableEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T,
                                         VERTEX_MAP_T>> {
 public:
  using traits_t = ImmutableEdgecutFragmentTraits<OID_T, VID_T, VDATA_T,
                                                  EDATA_T, VERTEX_MAP_T>;
  using base_t =
      CSREdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>;
  using internal_vertex_t = internal::Vertex<VID_T, VDATA_T>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_map_t = typename traits_t::vertex_map_t;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

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

  ImmutableEdgecutFragment() {}

  explicit ImmutableEdgecutFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>(vm_ptr) {}

  virtual ~ImmutableEdgecutFragment() = default;

  using base_t::buildCSR;
  using base_t::init;
  using base_t::IsInnerVertexGid;

  static std::string type_info() {
    std::string ret = "";
    if (std::is_same<EDATA_T, EmptyType>::value) {
      ret += "empty";
    } else if (std::is_same<EDATA_T, double>::value) {
      ret += "double";
    } else if (std::is_same<EDATA_T, float>::value) {
      ret += "float";
    } else {
      LOG(FATAL) << "Edge data type not supported...";
    }

    if (_load_strategy == LoadStrategy::kOnlyOut) {
      ret += "_out";
    } else if (_load_strategy == LoadStrategy::kOnlyIn) {
      ret += "_in";
    } else if (_load_strategy == LoadStrategy::kBothOutIn) {
      ret += "_both";
    } else {
      LOG(FATAL) << "Invalid load strategy...";
    }

    using partitioner_t = typename VERTEX_MAP_T::partitioner_t;
    if (std::is_same<partitioner_t, HashPartitioner<OID_T>>::value) {
      ret += "_hash";
    } else if (std::is_same<partitioner_t,
                            SegmentedPartitioner<OID_T>>::value) {
      ret += "_seg";
    }

    return ret;
  }

  void Init(fid_t fid, bool directed, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) override {
    init(fid, directed);

    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    {
      std::vector<VID_T> outer_vertices;
      auto iter_in = [&](Edge<VID_T, EDATA_T>& e,
                         std::vector<VID_T>& outer_vertices) {
        if (IsInnerVertexGid(e.dst)) {
          if (!IsInnerVertexGid(e.src)) {
            outer_vertices.push_back(e.src);
          }
        } else {
          e.src = invalid_vid;
        }
      };
      auto iter_out = [&](Edge<VID_T, EDATA_T>& e,
                          std::vector<VID_T>& outer_vertices) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            outer_vertices.push_back(e.dst);
          }
        } else {
          e.src = invalid_vid;
        }
      };
      auto iter_out_in = [&](Edge<VID_T, EDATA_T>& e,
                             std::vector<VID_T>& outer_vertices) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            outer_vertices.push_back(e.dst);
          }
        } else if (IsInnerVertexGid(e.dst)) {
          outer_vertices.push_back(e.src);
        } else {
          e.src = invalid_vid;
        }
      };

      auto iter_in_undirected = [&](Edge<VID_T, EDATA_T>& e,
                                    std::vector<VID_T>& outer_vertices) {
        if (IsInnerVertexGid(e.dst)) {
          if (!IsInnerVertexGid(e.src)) {
            outer_vertices.push_back(e.src);
          }
        } else {
          if (IsInnerVertexGid(e.src)) {
            outer_vertices.push_back(e.dst);
          } else {
            e.src = invalid_vid;
          }
        }
      };
      auto iter_out_undirected = [&](Edge<VID_T, EDATA_T>& e,
                                     std::vector<VID_T>& outer_vertices) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            outer_vertices.push_back(e.dst);
          }
        } else {
          if (IsInnerVertexGid(e.dst)) {
            outer_vertices.push_back(e.src);
          } else {
            e.src = invalid_vid;
          }
        }
      };

      if (load_strategy == LoadStrategy::kOnlyIn) {
        if (directed) {
          for (auto& e : edges) {
            iter_in(e, outer_vertices);
          }
        } else {
          for (auto& e : edges) {
            iter_in_undirected(e, outer_vertices);
          }
        }
      } else if (load_strategy == LoadStrategy::kOnlyOut) {
        if (directed) {
          for (auto& e : edges) {
            iter_out(e, outer_vertices);
          }
        } else {
          for (auto& e : edges) {
            iter_out_undirected(e, outer_vertices);
          }
        }
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (auto& e : edges) {
          iter_out_in(e, outer_vertices);
        }
      } else {
        LOG(FATAL) << "Invalid load strategy";
      }

      DistinctSort(outer_vertices);

      ovgid_.resize(outer_vertices.size());
      memcpy(&ovgid_[0], &outer_vertices[0],
             outer_vertices.size() * sizeof(VID_T));
    }

    vid_t ovid = ivnum_;
    for (auto gid : ovgid_) {
      ovg2l_.emplace(gid, ovid);
      ++ovid;
    }
    ovnum_ = ovid - ivnum_;
    this->inner_vertices_.SetRange(0, ivnum_);
    this->outer_vertices_.SetRange(ivnum_, ivnum_ + ovnum_);
    this->vertices_.SetRange(0, ivnum_ + ovnum_);

    buildCSR(this->Vertices(), edges, load_strategy);

    initOuterVerticesOfFragment();

    vdata_.clear();
    vdata_.resize(ivnum_ + ovnum_);
    if (sizeof(internal_vertex_t) > sizeof(VID_T)) {
      for (auto& v : vertices) {
        VID_T gid = v.vid;
        if (id_parser_.get_fragment_id(gid) == fid_) {
          vdata_[id_parser_.get_local_id(gid)] = v.vdata;
        } else {
          auto iter = ovg2l_.find(gid);
          if (iter != ovg2l_.end()) {
            vdata_[iter->second] = v.vdata;
          }
        }
      }
    }
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid_);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    base_t::serialize(io_adaptor);

    InArchive ia;

    int ils = underlying_value(load_strategy);
    ia << ovnum_ << ils;
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();

    if (ovnum_ > 0) {
      CHECK(io_adaptor->Write(&ovgid_[0], ovnum_ * sizeof(VID_T)));
    }

    ia << vdata_;
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();

    io_adaptor->Close();
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& prefix, const fid_t fid) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid);
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    base_t::deserialize(io_adaptor);

    OutArchive oa;
    int ils;
    CHECK(io_adaptor->ReadArchive(oa));
    oa >> ovnum_ >> ils;
    auto got_load_strategy = LoadStrategy(ils);
    if (got_load_strategy != load_strategy) {
      LOG(FATAL) << "load strategy not consistent.";
    }

    oa.Clear();

    ovgid_.clear();
    ovgid_.resize(ovnum_);
    if (ovnum_ > 0) {
      CHECK(io_adaptor->Read(&ovgid_[0], ovnum_ * sizeof(VID_T)));
    }

    initOuterVerticesOfFragment();

    {
      ovg2l_.clear();
      VID_T ovid = ivnum_;
      for (auto gid : ovgid_) {
        ovg2l_.emplace(gid, ovid);
        ++ovid;
      }
    }

    CHECK(io_adaptor->ReadArchive(oa));
    oa >> vdata_;

    io_adaptor->Close();
  }

  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) override {
    base_t::PrepareToRunApp(comm_spec, conf);
    if (conf.need_split_edges_by_fragment && !splited_edges_by_fragment_) {
      splitEdgesByFragment();
      splited_edges_by_fragment_ = true;
    } else if (conf.need_split_edges && !splited_edges_) {
      splitEdges();
      splited_edges_ = true;
    }
  }

  using base_t::IsInnerVertex;

  using base_t::GetFragId;

  inline const VDATA_T& GetData(const vertex_t& v) const override {
    return vdata_[v.GetValue()];
  }

  inline void SetData(const vertex_t& v, const VDATA_T& val) override {
    vdata_[v.GetValue()] = val;
  }

  bool OuterVertexGid2Lid(VID_T gid, VID_T& lid) const override {
    auto iter = ovg2l_.find(gid);
    if (iter != ovg2l_.end()) {
      lid = iter->second;
      return true;
    } else {
      return false;
    }
  }

  VID_T GetOuterVertexGid(vertex_t v) const override {
    return ovgid_[v.GetValue() - ivnum_];
  }

 public:
  using base_t::GetIncomingAdjList;
  using base_t::GetOutgoingAdjList;

 protected:
  using base_t::get_ie_begin;
  using base_t::get_ie_end;
  using base_t::get_oe_begin;
  using base_t::get_oe_end;

 public:
  using adj_list_t = typename base_t::adj_list_t;
  using const_adj_list_t = typename base_t::const_adj_list_t;
  inline adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return adj_list_t(get_ie_begin(v), iespliters_[0][v]);
  }

  inline const_adj_list_t GetIncomingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return const_adj_list_t(get_ie_begin(v), iespliters_[0][v]);
  }

  inline adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return adj_list_t(iespliters_[0][v], get_ie_end(v));
  }

  inline const_adj_list_t GetIncomingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    return const_adj_list_t(iespliters_[0][v], get_ie_end(v));
  }

  inline adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return adj_list_t(get_oe_begin(v), oespliters_[0][v]);
  }

  inline const_adj_list_t GetOutgoingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return const_adj_list_t(get_oe_begin(v), oespliters_[0][v]);
  }

  inline adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return adj_list_t(oespliters_[0][v], get_oe_end(v));
  }

  inline const_adj_list_t GetOutgoingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return const_adj_list_t(oespliters_[0][v], get_oe_end(v));
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v,
                                       fid_t src_fid) override {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    assert(src_fid != fid_);
    return adj_list_t(iespliters_[src_fid][v], iespliters_[src_fid + 1][v]);
  }

  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                             fid_t src_fid) const override {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    assert(src_fid != fid_);
    return const_adj_list_t(iespliters_[src_fid][v],
                            iespliters_[src_fid + 1][v]);
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                       fid_t dst_fid) override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    assert(dst_fid != fid_);
    return adj_list_t(oespliters_[dst_fid][v], oespliters_[dst_fid + 1][v]);
  }

  inline const_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                             fid_t dst_fid) const override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    assert(dst_fid != fid_);
    return const_adj_list_t(oespliters_[dst_fid][v],
                            oespliters_[dst_fid + 1][v]);
  }

 protected:
  void initOuterVerticesOfFragment() {
    std::vector<int> frag_v_num(fnum_, 0);
    fid_t cur_fid = 0;
    for (VID_T i = 0; i < ovnum_; ++i) {
      fid_t fid = id_parser_.get_fragment_id(ovgid_[i]);
      CHECK_GE(fid, cur_fid);
      cur_fid = fid;
      ++frag_v_num[fid];
    }
    outer_vertices_of_frag_.clear();
    outer_vertices_of_frag_.reserve(fnum_);
    VID_T cur_lid = ivnum_;
    for (fid_t i = 0; i < fnum_; ++i) {
      VID_T next_lid = cur_lid + frag_v_num[i];
      outer_vertices_of_frag_.emplace_back(cur_lid, next_lid);
      cur_lid = next_lid;
    }
    CHECK_EQ(cur_lid, ivnum_ + ovnum_);
  }

  void splitEdges() {
    auto inner_vertices = base_t::InnerVertices();
    iespliters_.clear();
    iespliters_.resize(1);
    oespliters_.clear();
    oespliters_.resize(1);
    iespliters_[0].Init(inner_vertices);
    oespliters_[0].Init(inner_vertices);
#ifndef PARALLEL_PREPARE
    int inner_neighbor_count = 0;
    for (auto& v : inner_vertices) {
      inner_neighbor_count = 0;
      auto ie = GetIncomingAdjList(v);
      for (auto& e : ie) {
        if (IsInnerVertex(e.neighbor)) {
          ++inner_neighbor_count;
        }
      }
      iespliters_[0][v] = get_ie_begin(v) + inner_neighbor_count;

      inner_neighbor_count = 0;
      auto oe = GetOutgoingAdjList(v);
      for (auto& e : oe) {
        if (IsInnerVertex(e.neighbor)) {
          ++inner_neighbor_count;
        }
      }
      oespliters_[0][v] = get_oe_begin(v) + inner_neighbor_count;
    }
#else
    int concurrency = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < concurrency; ++i) {
      threads.emplace_back(std::thread(
          [&](int tid) {
            VID_T batch = (ivnum_ + concurrency - 1) / concurrency;
            VID_T from = std::min(batch * tid, ivnum_);
            VID_T to = std::min(from + batch, ivnum_);

            vertex_t v(from);
            int inner_neighbor_count = 0;
            while (v.GetValue() != to) {
              inner_neighbor_count = 0;
              auto ie = GetIncomingAdjList(v);
              for (auto& e : ie) {
                if (IsInnerVertex(e.neighbor)) {
                  ++inner_neighbor_count;
                }
              }
              iespliters_[0][v] = get_ie_begin(v) + inner_neighbor_count;

              inner_neighbor_count = 0;
              auto oe = GetOutgoingAdjList(v);
              for (auto& e : oe) {
                if (IsInnerVertex(e.neighbor)) {
                  ++inner_neighbor_count;
                }
              }
              oespliters_[0][v] = get_oe_begin(v) + inner_neighbor_count;
              ++v;
            }
          },
          i));
    }
    for (auto& thrd : threads) {
      thrd.join();
    }
#endif
  }

  void splitEdgesByFragment() {
    auto inner_vertices = base_t::InnerVertices();
    iespliters_.clear();
    iespliters_.resize(fnum_ + 1);
    oespliters_.clear();
    oespliters_.resize(fnum_ + 1);
    for (fid_t i = 0; i < fnum_ + 1; ++i) {
      iespliters_[i].Init(inner_vertices);
      oespliters_[i].Init(inner_vertices);
    }
#ifndef PARALLEL_PREPARE
    std::vector<int> frag_count(fnum_, 0);
    for (auto& v : inner_vertices) {
      auto ie = GetIncomingAdjList(v);
      for (auto& e : ie) {
        ++frag_count[GetFragId(e.neighbor)];
      }
      iespliters_[0][v] = get_ie_begin(v) + frag_count[fid_];
      frag_count[fid_] = 0;
      for (fid_t j = 0; j < fnum_; ++j) {
        iespliters_[j + 1][v] = iespliters_[j][v] + frag_count[j];
        frag_count[j] = 0;
      }

      auto oe = GetOutgoingAdjList(v);
      for (auto& e : oe) {
        ++frag_count[GetFragId(e.neighbor)];
      }
      oespliters_[0][v] = get_oe_begin(v) + frag_count[fid_];
      frag_count[fid_] = 0;
      for (fid_t j = 0; j < fnum_; ++j) {
        oespliters_[j + 1][v] = oespliters_[j][v] + frag_count[j];
        frag_count[j] = 0;
      }
    }
#else
    int concurrency = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < concurrency; ++i) {
      threads.emplace_back(std::thread(
          [&](int tid) {
            VID_T batch = (ivnum_ + concurrency - 1) / concurrency;
            VID_T from = std::min(batch * tid, ivnum_);
            VID_T to = std::min(from + batch, ivnum_);

            vertex_t v(from);
            std::vector<int> frag_count(fnum_, 0);
            while (v.GetValue() != to) {
              auto ie = GetIncomingAdjList(v);
              for (auto& e : ie) {
                ++frag_count[GetFragId(e.neighbor)];
              }
              iespliters_[0][v] = get_ie_begin(v) + frag_count[fid_];
              frag_count[fid_] = 0;
              for (fid_t j = 0; j < fnum_; ++j) {
                iespliters_[j + 1][v] = iespliters_[j][v] + frag_count[j];
                frag_count[j] = 0;
              }

              auto oe = GetOutgoingAdjList(v);
              for (auto& e : oe) {
                ++frag_count[GetFragId(e.neighbor)];
              }
              oespliters_[0][v] = get_oe_begin(v) + frag_count[fid_];
              frag_count[fid_] = 0;
              for (fid_t j = 0; j < fnum_; ++j) {
                oespliters_[j + 1][v] = oespliters_[j][v] + frag_count[j];
                frag_count[j] = 0;
              }
              ++v;
            }
          },
          i));
    }
    for (auto& thrd : threads) {
      thrd.join();
    }
#endif
  }

  using base_t::ivnum_;
  VID_T ovnum_;
  using base_t::directed_;
  using base_t::fid_;
  using base_t::fnum_;
  using base_t::id_parser_;

  ska::flat_hash_map<VID_T, VID_T, std::hash<VID_T>, std::equal_to<VID_T>,
                     Allocator<std::pair<VID_T, VID_T>>>
      ovg2l_;
  Array<VID_T, Allocator<VID_T>> ovgid_;
  Array<VDATA_T, Allocator<VDATA_T>> vdata_;

  using base_t::outer_vertices_of_frag_;

  std::vector<VertexArray<inner_vertices_t, nbr_t*>> iespliters_, oespliters_;
  bool splited_edges_by_fragment_ = false;
  bool splited_edges_ = false;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_IMMUTABLE_EDGECUT_FRAGMENT_H_
