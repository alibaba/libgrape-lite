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
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/config.h"
#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
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
          LoadStrategy _load_strategy = LoadStrategy::kOnlyOut>
class ImmutableEdgecutFragment
    : public EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T> {
 public:
  using internal_vertex_t = internal::Vertex<VID_T, VDATA_T>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  template <typename DATA_T>
  using vertex_array_t = VertexArray<DATA_T, vid_t>;
  using vertex_map_t = GlobalVertexMap<oid_t, vid_t>;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

  static constexpr LoadStrategy load_strategy = _load_strategy;

  ImmutableEdgecutFragment() = default;

  explicit ImmutableEdgecutFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : vm_ptr_(vm_ptr) {}

  virtual ~ImmutableEdgecutFragment() = default;

  void Init(fid_t fid, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) override {
    fid_ = fid;
    fnum_ = vm_ptr_->GetFragmentNum();
    calcFidBitWidth(fnum_, id_mask_, fid_offset_);

    ivnum_ = vm_ptr_->GetInnerVertexSize(fid);

    tvnum_ = ivnum_;
    oenum_ = 0;
    ienum_ = 0;

    ie_.clear();
    oe_.clear();
    ieoffset_.clear();
    oeoffset_.clear();

    VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    auto is_iv_gid = [this](VID_T id) { return (id >> fid_offset_) == fid_; };
    {
      std::vector<VID_T> outer_vertices;
      auto first_iter_in = [&is_iv_gid, invalid_vid](
                               Edge<VID_T, EDATA_T>& e,
                               std::vector<VID_T>& outer_vertices) {
        if (is_iv_gid(e.dst_)) {
          if (!is_iv_gid(e.src_)) {
            outer_vertices.push_back(e.src_);
          }
        } else {
          e.src_ = invalid_vid;
        }
      };
      auto first_iter_out = [&is_iv_gid, invalid_vid](
                                Edge<VID_T, EDATA_T>& e,
                                std::vector<VID_T>& outer_vertices) {
        if (is_iv_gid(e.src_)) {
          if (!is_iv_gid(e.dst_)) {
            outer_vertices.push_back(e.dst_);
          }
        } else {
          e.src_ = invalid_vid;
        }
      };
      auto first_iter_out_in = [&is_iv_gid, invalid_vid](
                                   Edge<VID_T, EDATA_T>& e,
                                   std::vector<VID_T>& outer_vertices) {
        if (is_iv_gid(e.src_)) {
          if (!is_iv_gid(e.dst_)) {
            outer_vertices.push_back(e.dst_);
          }
        } else if (is_iv_gid(e.dst_)) {
          outer_vertices.push_back(e.src_);
        } else {
          e.src_ = invalid_vid;
        }
      };

      if (load_strategy == LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          first_iter_in(e, outer_vertices);
        }
      } else if (load_strategy == LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          first_iter_out(e, outer_vertices);
        }
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (auto& e : edges) {
          first_iter_out_in(e, outer_vertices);
        }
      } else {
        LOG(FATAL) << "Invalid load strategy";
      }

      DistinctSort(outer_vertices);

      ovgid_.resize(outer_vertices.size());
      memcpy(&ovgid_[0], &outer_vertices[0],
             outer_vertices.size() * sizeof(VID_T));
    }

    tvnum_ = ivnum_;
    for (auto gid : ovgid_) {
      ovg2l_.emplace(gid, tvnum_);
      ++tvnum_;
    }
    ovnum_ = tvnum_ - ivnum_;

    {
      std::vector<int> idegree(tvnum_, 0), odegree(tvnum_, 0);
      ienum_ = 0;
      oenum_ = 0;

      auto gid_to_lid = [this](VID_T gid) {
        return ((gid >> fid_offset_) == fid_) ? (gid & id_mask_)
                                              : (ovg2l_.at(gid));
      };

      auto iv_gid_to_lid = [this](VID_T gid) { return gid & id_mask_; };
      auto ov_gid_to_lid = [this](VID_T gid) { return ovg2l_.at(gid); };

      auto second_iter_in = [this, &iv_gid_to_lid, &ov_gid_to_lid, invalid_vid,
                             &is_iv_gid](Edge<VID_T, EDATA_T>& e,
                                         std::vector<int>& idegree,
                                         std::vector<int>& odegree) {
        if (e.src_ != invalid_vid) {
          if (is_iv_gid(e.src_)) {
            e.src_ = iv_gid_to_lid(e.src_);
          } else {
            e.src_ = ov_gid_to_lid(e.src_);
            ++odegree[e.src_];
            ++oenum_;
          }
          e.dst_ = iv_gid_to_lid(e.dst_);
          ++idegree[e.dst_];
          ++ienum_;
        }
      };

      auto second_iter_out = [this, &iv_gid_to_lid, &ov_gid_to_lid, invalid_vid,
                              &is_iv_gid](Edge<VID_T, EDATA_T>& e,
                                          std::vector<int>& idegree,
                                          std::vector<int>& odegree) {
        if (e.src_ != invalid_vid) {
          e.src_ = iv_gid_to_lid(e.src_);
          if (is_iv_gid(e.dst_)) {
            e.dst_ = iv_gid_to_lid(e.dst_);
          } else {
            e.dst_ = ov_gid_to_lid(e.dst_);
            ++idegree[e.dst_];
            ++ienum_;
          }
          ++odegree[e.src_];
          ++oenum_;
        }
      };

      auto second_iter_out_in = [this, &gid_to_lid, invalid_vid](
                                    Edge<VID_T, EDATA_T>& e,
                                    std::vector<int>& idegree,
                                    std::vector<int>& odegree) {
        if (e.src_ != invalid_vid) {
          e.src_ = gid_to_lid(e.src_);
          e.dst_ = gid_to_lid(e.dst_);
          ++odegree[e.src_];
          ++idegree[e.dst_];
          ++oenum_;
          ++ienum_;
        }
      };

      if (load_strategy == LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          second_iter_in(e, idegree, odegree);
        }
      } else if (load_strategy == LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          second_iter_out(e, idegree, odegree);
        }
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (auto& e : edges) {
          second_iter_out_in(e, idegree, odegree);
        }
      } else {
        LOG(FATAL) << "Invalid load strategy";
      }

      ie_.resize(ienum_);
      oe_.resize(oenum_);
      ieoffset_.resize(tvnum_ + 1);
      oeoffset_.resize(tvnum_ + 1);
      ieoffset_[0] = &ie_[0];
      oeoffset_[0] = &oe_[0];

      for (VID_T i = 0; i < tvnum_; ++i) {
        ieoffset_[i + 1] = ieoffset_[i] + idegree[i];
        oeoffset_[i + 1] = oeoffset_[i] + odegree[i];
      }
    }

    {
      Array<nbr_t*, Allocator<nbr_t*>> ieiter(ieoffset_), oeiter(oeoffset_);

      auto third_iter_in = [invalid_vid, this](
                               const Edge<VID_T, EDATA_T>& e,
                               Array<nbr_t*, Allocator<nbr_t*>>& ieiter,
                               Array<nbr_t*, Allocator<nbr_t*>>& oeiter) {
        if (e.src_ != invalid_vid) {
          ieiter[e.dst_]->GetEdgeSrc(e);
          ++ieiter[e.dst_];
          if (e.src_ >= ivnum_) {
            oeiter[e.src_]->GetEdgeDst(e);
            ++oeiter[e.src_];
          }
        }
      };

      auto third_iter_out = [invalid_vid, this](
                                const Edge<VID_T, EDATA_T>& e,
                                Array<nbr_t*, Allocator<nbr_t*>>& ieiter,
                                Array<nbr_t*, Allocator<nbr_t*>>& oeiter) {
        if (e.src_ != invalid_vid) {
          oeiter[e.src_]->GetEdgeDst(e);
          ++oeiter[e.src_];
          if (e.dst_ >= ivnum_) {
            ieiter[e.dst_]->GetEdgeSrc(e);
            ++ieiter[e.dst_];
          }
        }
      };

      auto third_iter_out_in = [invalid_vid](
                                   const Edge<VID_T, EDATA_T>& e,
                                   Array<nbr_t*, Allocator<nbr_t*>>& ieiter,
                                   Array<nbr_t*, Allocator<nbr_t*>>& oeiter) {
        if (e.src_ != invalid_vid) {
          ieiter[e.dst_]->GetEdgeSrc(e);
          ++ieiter[e.dst_];
          oeiter[e.src_]->GetEdgeDst(e);
          ++oeiter[e.src_];
        }
      };

      if (load_strategy == LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          third_iter_in(e, ieiter, oeiter);
        }
      } else if (load_strategy == LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          third_iter_out(e, ieiter, oeiter);
        }
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (auto& e : edges) {
          third_iter_out_in(e, ieiter, oeiter);
        }
      } else {
        LOG(FATAL) << "Invalid load strategy";
      }
    }

    for (VID_T i = 0; i < tvnum_; ++i) {
      std::sort(ieoffset_[i], ieoffset_[i + 1],
                [](const nbr_t& lhs, const nbr_t& rhs) {
                  return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
    }
    for (VID_T i = 0; i < tvnum_; ++i) {
      std::sort(oeoffset_[i], oeoffset_[i + 1],
                [](const nbr_t& lhs, const nbr_t& rhs) {
                  return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
    }

    initOuterVerticesOfFragment();

    vdata_.clear();
    vdata_.resize(tvnum_);
    if (sizeof(internal_vertex_t) > sizeof(VID_T)) {
      for (auto& v : vertices) {
        VID_T gid = v.vid();
        if (gid >> fid_offset_ == fid_) {
          vdata_[(gid & id_mask_)] = v.vdata();
        } else {
          auto iter = ovg2l_.find(gid);
          if (iter != ovg2l_.end()) {
            vdata_[iter->second] = v.vdata();
          }
        }
      }
    }

    mirrors_range_.resize(fnum_);
    mirrors_range_[fid_].SetRange(0, 0);
    mirrors_of_frag_.resize(fnum_);
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid_);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    InArchive ia;

    io_adaptor->Open("wb");

    int ils = underlying_value(load_strategy);
    ia << ivnum_ << ovnum_ << ienum_ << oenum_ << fid_ << fnum_ << ils;
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();

    if (ovnum_ > 0) {
      CHECK(io_adaptor->Write(&ovgid_[0], ovnum_ * sizeof(VID_T)));
    }

    {
      if (std::is_pod<EDATA_T>::value || (sizeof(nbr_t) == sizeof(VID_T))) {
        if (ienum_ > 0) {
          CHECK(io_adaptor->Write(&ie_[0], ienum_ * sizeof(nbr_t)));
        }
        if (oenum_ > 0) {
          CHECK(io_adaptor->Write(&oe_[0], oenum_ * sizeof(nbr_t)));
        }
      } else {
        ia << ie_;
        CHECK(io_adaptor->WriteArchive(ia));
        ia.Clear();

        ia << oe_;
        CHECK(io_adaptor->WriteArchive(ia));
        ia.Clear();
      }

      std::vector<int> idegree(tvnum_);
      for (VID_T i = 0; i < tvnum_; ++i) {
        idegree[i] = ieoffset_[i + 1] - ieoffset_[i];
      }
      CHECK(io_adaptor->Write(&idegree[0], sizeof(int) * tvnum_));

      std::vector<int> odegree(tvnum_);
      for (VID_T i = 0; i < tvnum_; ++i) {
        odegree[i] = oeoffset_[i + 1] - oeoffset_[i];
      }
      CHECK(io_adaptor->Write(&odegree[0], sizeof(int) * tvnum_));
    }

    for (fid_t i = 0; i < fnum_; ++i) {
      ia << mirrors_range_[i].begin().GetValue()
         << mirrors_range_[i].end().GetValue();
    }
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();

    for (fid_t i = 0; i < fnum_; ++i) {
      CHECK_EQ(mirrors_range_[i].size(), mirrors_of_frag_[i].size());
      if (mirrors_range_[i].size() != 0) {
        CHECK(io_adaptor->Write(&mirrors_of_frag_[i][0],
                                sizeof(vertex_t) * mirrors_of_frag_[i].size()));
      }
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

    OutArchive oa;
    int ils;
    CHECK(io_adaptor->ReadArchive(oa));
    oa >> ivnum_ >> ovnum_ >> ienum_ >> oenum_ >> fid_ >> fnum_ >> ils;
    auto got_load_strategy = LoadStrategy(ils);
    if (got_load_strategy != load_strategy) {
      LOG(FATAL) << "load strategy not consistent.";
    }
    tvnum_ = ivnum_ + ovnum_;
    calcFidBitWidth(fnum_, id_mask_, fid_offset_);

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

    ie_.clear();
    oe_.clear();
    if (std::is_pod<EDATA_T>::value || (sizeof(nbr_t) == sizeof(VID_T))) {
      ie_.resize(ienum_);
      if (ienum_ > 0) {
        CHECK(io_adaptor->Read(&ie_[0], ienum_ * sizeof(nbr_t)));
      }
      oe_.resize(oenum_);
      if (oenum_ > 0) {
        CHECK(io_adaptor->Read(&oe_[0], oenum_ * sizeof(nbr_t)));
      }
    } else {
      CHECK(io_adaptor->ReadArchive(oa));
      oa >> ie_;
      oa.Clear();
      CHECK_EQ(ie_.size(), ienum_);

      CHECK(io_adaptor->ReadArchive(oa));
      oa >> oe_;
      oa.Clear();
      CHECK_EQ(oe_.size(), oenum_);
    }

    ieoffset_.clear();
    ieoffset_.resize(tvnum_ + 1);
    ieoffset_[0] = &ie_[0];
    {
      std::vector<int> idegree(tvnum_);
      CHECK(io_adaptor->Read(&idegree[0], sizeof(int) * tvnum_));
      for (VID_T i = 0; i < tvnum_; ++i) {
        ieoffset_[i + 1] = ieoffset_[i] + idegree[i];
      }
    }

    oeoffset_.clear();
    oeoffset_.resize(tvnum_ + 1);
    oeoffset_[0] = &oe_[0];
    {
      std::vector<int> odegree(tvnum_);
      CHECK(io_adaptor->Read(&odegree[0], sizeof(int) * tvnum_));
      for (VID_T i = 0; i < tvnum_; ++i) {
        oeoffset_[i + 1] = oeoffset_[i] + odegree[i];
      }
    }

    mirrors_range_.clear();
    mirrors_range_.resize(fnum_);
    mirrors_of_frag_.clear();
    mirrors_of_frag_.resize(fnum_);
    CHECK(io_adaptor->ReadArchive(oa));
    for (fid_t i = 0; i < fnum_; ++i) {
      VID_T begin, end;
      oa >> begin >> end;
      mirrors_range_[i].SetRange(begin, end);
      VID_T len = end - begin;
      mirrors_of_frag_[i].resize(len);
      if (len != 0) {
        CHECK(
            io_adaptor->Read(&mirrors_of_frag_[i][0], len * sizeof(vertex_t)));
      }
    }
    oa.Clear();

    CHECK(io_adaptor->ReadArchive(oa));
    oa >> vdata_;

    io_adaptor->Close();
  }

  void PrepareToRunApp(MessageStrategy strategy,
                       bool need_split_edges) override {
    if (strategy == MessageStrategy::kAlongEdgeToOuterVertex ||
        strategy == MessageStrategy::kAlongIncomingEdgeToOuterVertex ||
        strategy == MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initMessageDestination(strategy);
    }

    if (need_split_edges) {
      initEdgesSplitter(ieoffset_, iespliters_);
      initEdgesSplitter(oeoffset_, oespliters_);
    }
  }

  inline fid_t fid() const override { return fid_; }

  inline fid_t fnum() const override { return fnum_; }

  inline VID_T id_mask() const { return id_mask_; }

  inline int fid_offset() const { return fid_offset_; }

  inline const vid_t* GetOuterVerticesGid() const { return &ovgid_[0]; }

  inline size_t GetEdgeNum() const override { return ienum_ + oenum_; }

  inline VID_T GetVerticesNum() const override { return tvnum_; }

  size_t GetTotalVerticesNum() const override {
    return vm_ptr_->GetTotalVertexSize();
  }

  inline VertexRange<VID_T> Vertices() const override {
    return VertexRange<VID_T>(0, tvnum_);
  }

  inline VertexRange<VID_T> InnerVertices() const override {
    return VertexRange<VID_T>(0, ivnum_);
  }

  inline VertexRange<VID_T> OuterVertices() const override {
    return VertexRange<VID_T>(ivnum_, tvnum_);
  }

  inline VertexRange<VID_T> OuterVertices(fid_t fid) const {
    return outer_vertices_of_frag_[fid];
  }

  inline bool GetVertex(const OID_T& oid, vertex_t& v) const override {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      return ((gid >> fid_offset_) == fid_) ? InnerVertexGid2Vertex(gid, v)
                                            : OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  inline OID_T GetId(const vertex_t& v) const override {
    return IsInnerVertex(v) ? GetInnerVertexId(v) : GetOuterVertexId(v);
  }

  inline fid_t GetFragId(const vertex_t& u) const override {
    return IsInnerVertex(u)
               ? fid_
               : (fid_t)(ovgid_[u.GetValue() - ivnum_] >> fid_offset_);
  }

  inline const VDATA_T& GetData(const vertex_t& v) const override {
    return vdata_[v.GetValue()];
  }

  inline void SetData(const vertex_t& v, const VDATA_T& val) override {
    vdata_[v.GetValue()] = val;
  }

  inline bool HasChild(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return oeoffset_[v.GetValue()] != oeoffset_[v.GetValue() + 1];
  }

  inline bool HasParent(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return ieoffset_[v.GetValue()] != ieoffset_[v.GetValue() + 1];
  }

  inline int GetLocalOutDegree(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return oeoffset_[v.GetValue() + 1] - oeoffset_[v.GetValue()];
  }

  inline int GetLocalInDegree(const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return ieoffset_[v.GetValue() + 1] - ieoffset_[v.GetValue()];
  }

  inline bool Gid2Vertex(const VID_T& gid, vertex_t& v) const override {
    return ((gid >> fid_offset_) == fid_) ? InnerVertexGid2Vertex(gid, v)
                                          : OuterVertexGid2Vertex(gid, v);
  }

  inline VID_T Vertex2Gid(const vertex_t& v) const override {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

  inline VID_T GetInnerVerticesNum() const override { return ivnum_; }

  inline VID_T GetOuterVerticesNum() const override { return ovnum_; }

  inline bool IsInnerVertex(const vertex_t& v) const override {
    return (v.GetValue() < ivnum_);
  }

  inline bool IsOuterVertex(const vertex_t& v) const override {
    return (v.GetValue() < tvnum_ && v.GetValue() >= ivnum_);
  }

  inline bool GetInnerVertex(const OID_T& oid, vertex_t& v) const override {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      if ((gid >> fid_offset_) == fid_) {
        v.SetValue(gid & id_mask_);
        return true;
      }
    }
    return false;
  }

  inline bool GetOuterVertex(const OID_T& oid, vertex_t& v) const override {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      return OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  inline OID_T GetInnerVertexId(const vertex_t& v) const override {
    OID_T internal_oid;
    vm_ptr_->GetOid(fid_, v.GetValue(), internal_oid);
    return internal_oid;
  }

  inline OID_T GetOuterVertexId(const vertex_t& v) const override {
    VID_T gid = ovgid_[v.GetValue() - ivnum_];
    OID_T internal_oid;
    vm_ptr_->GetOid(gid, internal_oid);
    return internal_oid;
  }

  inline OID_T Gid2Oid(const VID_T& gid) const {
    OID_T internal_oid;
    vm_ptr_->GetOid(gid, internal_oid);
    return internal_oid;
  }

  inline bool Oid2Gid(const OID_T& oid, VID_T& gid) const {
    OID_T internal_oid(oid);
    return vm_ptr_->GetGid(internal_oid, gid);
  }

  inline bool InnerVertexGid2Vertex(const VID_T& gid,
                                    vertex_t& v) const override {
    v.SetValue(gid & id_mask_);
    return true;
  }

  inline bool OuterVertexGid2Vertex(const VID_T& gid,
                                    vertex_t& v) const override {
    auto iter = ovg2l_.find(gid);
    if (iter != ovg2l_.end()) {
      v.SetValue(iter->second);
      return true;
    } else {
      return false;
    }
  }

  inline VID_T GetOuterVertexGid(const vertex_t& v) const override {
    return ovgid_[v.GetValue() - ivnum_];
  }
  inline VID_T GetInnerVertexGid(const vertex_t& v) const override {
    return (v.GetValue() | ((VID_T) fid_ << fid_offset_));
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
    return (!idoffset_.empty() && IsInnerVertex(v) &&
            (idoffset_[v.GetValue()] != idoffset_[v.GetValue() + 1]));
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
    return (!odoffset_.empty() && IsInnerVertex(v) &&
            (odoffset_[v.GetValue()] != odoffset_[v.GetValue() + 1]));
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
    return (!iodoffset_.empty() && IsInnerVertex(v) &&
            iodoffset_[v.GetValue()] != iodoffset_[v.GetValue() + 1]);
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
  inline DestList OEDests(const vertex_t& v) const override {
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
  inline DestList IOEDests(const vertex_t& v) const override {
    assert(!iodoffset_.empty());
    assert(IsInnerVertex(v));
    return DestList(iodoffset_[v.GetValue()], iodoffset_[v.GetValue() + 1]);
  }

 public:
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
    return adj_list_t(ieoffset_[v.GetValue()], ieoffset_[v.GetValue() + 1]);
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
    return const_adj_list_t(ieoffset_[v.GetValue()],
                            ieoffset_[v.GetValue() + 1]);
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
    return adj_list_t(oeoffset_[v.GetValue()], oeoffset_[v.GetValue() + 1]);
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
    return const_adj_list_t(oeoffset_[v.GetValue()],
                            oeoffset_[v.GetValue() + 1]);
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
  inline adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) override {
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
  inline const_adj_list_t GetIncomingInnerVertexAdjList(
      const vertex_t& v) const override {
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
  inline adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) override {
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
  inline const_adj_list_t GetIncomingOuterVertexAdjList(
      const vertex_t& v) const override {
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
  inline adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) override {
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
  inline const_adj_list_t GetOutgoingInnerVertexAdjList(
      const vertex_t& v) const override {
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
  inline adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) override {
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
  inline const_adj_list_t GetOutgoingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    return const_adj_list_t(oespliters_[0][v.GetValue()],
                            oeoffset_[v.GetValue() + 1]);
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v, fid_t src_fid) {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    assert(src_fid != fid_);
    return adj_list_t(iespliters_[src_fid][v.GetValue()],
                      iespliters_[src_fid + 1][v.GetValue()]);
  }

  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                             fid_t src_fid) const {
    assert(IsInnerVertex(v));
    assert(!iespliters_.empty());
    assert(src_fid != fid_);
    return const_adj_list_t(iespliters_[src_fid][v.GetValue()],
                            iespliters_[src_fid + 1][v.GetValue()]);
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v, fid_t dst_fid) {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    assert(dst_fid != fid_);
    return adj_list_t(oespliters_[dst_fid][v.GetValue()],
                      oespliters_[dst_fid + 1][v.GetValue()]);
  }

  inline const_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                             fid_t dst_fid) const {
    assert(IsInnerVertex(v));
    assert(!oespliters_.empty());
    assert(dst_fid != fid_);
    return const_adj_list_t(oespliters_[dst_fid][v.GetValue()],
                            oespliters_[dst_fid + 1][v.GetValue()]);
  }

  inline const std::vector<vertex_t>& MirrorVertices(fid_t fid) const {
    return mirrors_of_frag_[fid];
  }

  inline const VertexRange<VID_T>& MirrorsRange(fid_t fid) const {
    return mirrors_range_[fid];
  }

  void SetupMirrorInfo(fid_t fid, const VertexRange<VID_T>& range,
                       const std::vector<VID_T>& gid_list) {
    mirrors_range_[fid].SetRange(range.begin().GetValue(),
                                 range.end().GetValue());
    auto& vertex_vec = mirrors_of_frag_[fid];
    vertex_vec.resize(gid_list.size());
    for (size_t i = 0; i < gid_list.size(); ++i) {
      CHECK_EQ(gid_list[i] >> fid_offset_, fid_);
      vertex_vec[i].SetValue(gid_list[i] & id_mask_);
    }
  }

 private:
  void initMessageDestination(const MessageStrategy& msg_strategy) {
    if (msg_strategy == MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initDestFidList(false, true, odst_, odoffset_);
    } else if (msg_strategy ==
               MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      initDestFidList(true, false, idst_, idoffset_);
    } else if (msg_strategy == MessageStrategy::kAlongEdgeToOuterVertex) {
      initDestFidList(true, true, iodst_, iodoffset_);
    }
  }

  void initDestFidList(bool in_edge, bool out_edge,
                       Array<fid_t, Allocator<fid_t>>& fid_list,
                       Array<fid_t*, Allocator<fid_t*>>& fid_list_offset) {
    if (!fid_list_offset.empty()) {
      return;
    }
    std::set<fid_t> dstset;
    std::vector<fid_t> tmp_fids;
    std::vector<int> id_num(ivnum_, 0);

    for (VID_T i = 0; i < ivnum_; ++i) {
      dstset.clear();
      if (in_edge) {
        nbr_t* ptr = ieoffset_[i];
        while (ptr != ieoffset_[i + 1]) {
          VID_T lid = ptr->neighbor.GetValue();
          if (lid >= ivnum_) {
            fid_t f = (ovgid_[lid - ivnum_] >> fid_offset_);
            dstset.insert(f);
          }
          ++ptr;
        }
      }
      if (out_edge) {
        nbr_t* ptr = oeoffset_[i];
        while (ptr != oeoffset_[i + 1]) {
          VID_T lid = ptr->neighbor.GetValue();
          if (lid >= ivnum_) {
            fid_t f = (ovgid_[lid - ivnum_] >> fid_offset_);
            dstset.insert(f);
          }
          ++ptr;
        }
      }
      id_num[i] = dstset.size();
      for (auto fid : dstset) {
        tmp_fids.push_back(fid);
      }
    }

    fid_list.resize(tmp_fids.size());
    fid_list_offset.resize(ivnum_ + 1);

    memcpy(&fid_list[0], &tmp_fids[0], sizeof(fid_t) * fid_list.size());
    fid_list_offset[0] = fid_list.data();
    for (VID_T i = 0; i < ivnum_; ++i) {
      fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
    }
  }

  void initEdgesSplitter(
      Array<nbr_t*, Allocator<nbr_t*>>& eoffset,
      std::vector<Array<nbr_t*, Allocator<nbr_t*>>>& espliters) {
    if (!espliters.empty()) {
      return;
    }
    espliters.resize(fnum_ + 1);
    for (auto& vec : espliters) {
      vec.resize(ivnum_);
    }
    std::vector<int> frag_count;
    for (VID_T i = 0; i < ivnum_; ++i) {
      frag_count.clear();
      frag_count.resize(fnum_, 0);
      adj_list_t edges(eoffset[i], eoffset[i + 1]);
      for (auto& e : edges) {
        if (e.neighbor.GetValue() >= ivnum_) {
          fid_t fid = (ovgid_[e.neighbor.GetValue() - ivnum_] >> fid_offset_);
          ++frag_count[fid];
        } else {
          ++frag_count[fid_];
        }
      }
      espliters[0][i] = eoffset[i] + frag_count[fid_];
      frag_count[fid_] = 0;
      for (fid_t j = 0; j < fnum_; ++j) {
        espliters[j + 1][i] = espliters[j][i] + frag_count[j];
      }
    }
  }

  void initOuterVerticesOfFragment() {
    std::vector<int> frag_v_num(fnum_, 0);
    fid_t cur_fid = 0;
    for (VID_T i = 0; i < ovnum_; ++i) {
      fid_t fid = (ovgid_[i] >> fid_offset_);
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
    CHECK_EQ(cur_lid, tvnum_);
  }

  template <typename T>
  void calcFidBitWidth(fid_t fnum, T& id_mask, int& fid_offset) {
    fid_t maxfid = fnum - 1;
    if (maxfid == 0) {
      fid_offset = (sizeof(T) * 8) - 1;
    } else {
      int i = 0;
      while (maxfid) {
        maxfid >>= 1;
        ++i;
      }
      fid_offset = (sizeof(T) * 8) - i;
    }
    id_mask = ((T) 1 << fid_offset) - (T) 1;
  }

  std::shared_ptr<vertex_map_t> vm_ptr_;
  VID_T ivnum_, ovnum_, tvnum_, id_mask_;
  size_t ienum_{}, oenum_{};
  int fid_offset_{};
  fid_t fid_{}, fnum_{};

  ska::flat_hash_map<VID_T, VID_T> ovg2l_;
  Array<VID_T, Allocator<VID_T>> ovgid_;
  Array<nbr_t, Allocator<nbr_t>> ie_, oe_;
  Array<nbr_t*, Allocator<nbr_t*>> ieoffset_, oeoffset_;
  Array<VDATA_T, Allocator<VDATA_T>> vdata_;

  std::vector<VertexRange<VID_T>> outer_vertices_of_frag_;

  std::vector<VertexRange<VID_T>> mirrors_range_;
  std::vector<std::vector<vertex_t>> mirrors_of_frag_;

  Array<fid_t, Allocator<fid_t>> idst_, odst_, iodst_;
  Array<fid_t*, Allocator<fid_t*>> idoffset_, odoffset_, iodoffset_;

  std::vector<Array<nbr_t*, Allocator<nbr_t*>>> iespliters_, oespliters_;

  template <typename _FRAG_T, typename _PARTITIONER_T, typename _IOADAPTOR_T,
            typename _Enable>
  friend class BasicFragmentLoader;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_IMMUTABLE_EDGECUT_FRAGMENT_H_
