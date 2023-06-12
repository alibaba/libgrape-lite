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

#ifndef GRAPE_FRAGMENT_MUTABLE_EDGECUT_FRAGMENT_H_
#define GRAPE_FRAGMENT_MUTABLE_EDGECUT_FRAGMENT_H_

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/fragment/basic_fragment_mutator.h"
#include "grape/fragment/csr_edgecut_fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/de_mutable_csr.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/types.h"
#include "grape/util.h"
#include "grape/vertex_map/global_vertex_map.h"

namespace grape {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          typename VERTEX_MAP_T>
struct MutableEdgecutFragmentTraits {
  using inner_vertices_t = VertexRange<VID_T>;
  using outer_vertices_t = VertexRange<VID_T>;
  using vertices_t = DualVertexRange<VID_T>;
  using sub_vertices_t = VertexVector<VID_T>;

  using fragment_adj_list_t =
      FilterAdjList<VID_T, EDATA_T,
                    std::function<bool(const Nbr<VID_T, EDATA_T>&)>>;
  using fragment_const_adj_list_t =
      FilterConstAdjList<VID_T, EDATA_T,
                         std::function<bool(const Nbr<VID_T, EDATA_T>&)>>;

  using csr_t = DeMutableCSR<VID_T, Nbr<VID_T, EDATA_T>>;
  using csr_builder_t = DeMutableCSRBuilder<VID_T, Nbr<VID_T, EDATA_T>>;
  using vertex_map_t = VERTEX_MAP_T;
  using mirror_vertices_t = std::vector<Vertex<VID_T>>;
};

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy _load_strategy = LoadStrategy::kOnlyOut,
          typename VERTEX_MAP_T =
              GlobalVertexMap<OID_T, VID_T, HashPartitioner<OID_T>>>
class MutableEdgecutFragment
    : public CSREdgecutFragmentBase<
          OID_T, VID_T, VDATA_T, EDATA_T,
          MutableEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T,
                                       VERTEX_MAP_T>> {
 public:
  using traits_t = MutableEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T,
                                                VERTEX_MAP_T>;
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

  using inner_vertices_t = typename traits_t::inner_vertices_t;
  using outer_vertices_t = typename traits_t::outer_vertices_t;
  using vertices_t = typename traits_t::vertices_t;
  using fragment_adj_list_t = typename traits_t::fragment_adj_list_t;
  using fragment_const_adj_list_t =
      typename traits_t::fragment_const_adj_list_t;

  template <typename T>
  using inner_vertex_array_t = VertexArray<inner_vertices_t, T>;

  template <typename T>
  using outer_vertex_array_t = VertexArray<outer_vertices_t, T>;

  template <typename T>
  using vertex_array_t = VertexArray<vertices_t, T>;

  explicit MutableEdgecutFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>(vm_ptr) {}
  virtual ~MutableEdgecutFragment() = default;

  using base_t::buildCSR;
  using base_t::init;
  using base_t::IsInnerVertexGid;

  static std::string type_info() { return ""; }

  void Init(fid_t fid, bool directed, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) override {
    init(fid, directed);

    ovnum_ = 0;
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    if (load_strategy == LoadStrategy::kOnlyIn) {
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.dst)) {
          if (!IsInnerVertexGid(e.src)) {
            parseOrAddOuterVertexGid(e.src);
          }
        } else {
          if (!directed && IsInnerVertexGid(e.src)) {
            parseOrAddOuterVertexGid(e.dst);
          } else {
            e.src = invalid_vid;
          }
        }
      }
    } else if (load_strategy == LoadStrategy::kOnlyOut) {
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.dst);
          }
        } else {
          if (!directed && IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.src);
          } else {
            e.src = invalid_vid;
          }
        }
      }
    } else if (load_strategy == LoadStrategy::kBothOutIn) {
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.dst);
          }
        } else {
          if (IsInnerVertexGid(e.dst)) {
            parseOrAddOuterVertexGid(e.src);
          } else {
            e.src = invalid_vid;
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid load strategy";
    }

    this->inner_vertices_.SetRange(0, ivnum_);
    this->outer_vertices_.SetRange(id_parser_.max_local_id() - ovnum_,
                                   id_parser_.max_local_id());
    this->vertices_.SetRange(0, ivnum_, id_parser_.max_local_id() - ovnum_,
                             id_parser_.max_local_id());
    initOuterVerticesOfFragment();

    buildCSR(this->Vertices(), edges, load_strategy);

    ivdata_.clear();
    ivdata_.resize(ivnum_);
    ovdata_.clear();
    ovdata_.resize(ovnum_);
    if (sizeof(internal_vertex_t) > sizeof(VID_T)) {
      for (auto& v : vertices) {
        VID_T gid = v.vid;
        if (id_parser_.get_fragment_id(gid) == fid_) {
          ivdata_[id_parser_.get_local_id(gid)] = v.vdata;
        } else {
          auto iter = ovg2i_.find(gid);
          if (iter != ovg2i_.end()) {
            ovdata_[outerVertexLidToIndex(iter->second)] = v.vdata;
          }
        }
      }
    }
  }

  using base_t::Gid2Lid;
  using base_t::ie_;
  using base_t::oe_;
  using base_t::vm_ptr_;
  void Mutate(Mutation<vid_t, vdata_t, edata_t>& mutation) {
    vertex_t v;
    if (!mutation.vertices_to_remove.empty() &&
        static_cast<double>(mutation.vertices_to_remove.size()) /
                static_cast<double>(this->GetVerticesNum()) <
            0.1) {
      std::set<vertex_t> sparse_set;
      for (auto gid : mutation.vertices_to_remove) {
        if (Gid2Vertex(gid, v)) {
          ie_.remove_vertex(v.GetValue());
          oe_.remove_vertex(v.GetValue());
          sparse_set.insert(v);
        }
      }
      if (!sparse_set.empty()) {
        auto func = [&sparse_set](vid_t i, const nbr_t& e) {
          return sparse_set.find(e.neighbor) != sparse_set.end();
        };
        ie_.remove_if(func);
        oe_.remove_if(func);
      }
    } else if (!mutation.vertices_to_remove.empty()) {
      vertex_array_t<bool> dense_bitset;
      for (auto gid : mutation.vertices_to_remove) {
        if (Gid2Vertex(gid, v)) {
          ie_.remove_vertex(v.GetValue());
          oe_.remove_vertex(v.GetValue());
          dense_bitset[v] = true;
        }
      }
      auto func = [&dense_bitset](vid_t i, const nbr_t& e) {
        return dense_bitset[e.neighbor];
      };
      ie_.remove_if(func);
      oe_.remove_if(func);
    }
    {
      static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
      for (auto& e : mutation.edges_to_remove) {
        if (!(Gid2Lid(e.first, e.first) && Gid2Lid(e.second, e.second))) {
          e.first = sentinel;
        }
      }
      ie_.remove_reversed_edges(mutation.edges_to_remove);
      oe_.remove_edges(mutation.edges_to_remove);
    }
    {
      static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
      for (auto& e : mutation.edges_to_update) {
        if (!(Gid2Lid(e.src, e.src) && Gid2Lid(e.dst, e.dst))) {
          e.src = sentinel;
        }
      }
      ie_.update_reversed_edges(mutation.edges_to_update);
      oe_.update_edges(mutation.edges_to_update);
    }
    {
      vid_t ivnum = this->GetInnerVerticesNum();
      vid_t ovnum = this->GetOuterVerticesNum();
      auto& edges_to_add = mutation.edges_to_add;
      static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
      if (load_strategy == LoadStrategy::kOnlyIn) {
        for (auto& e : edges_to_add) {
          if (IsInnerVertexGid(e.dst)) {
            e.dst = id_parser_.get_local_id(e.dst);
            if (!IsInnerVertexGid(e.src)) {
              e.src = parseOrAddOuterVertexGid(e.src);
            } else {
              e.src = id_parser_.get_local_id(e.src);
            }
          } else {
            if (!directed_ && IsInnerVertexGid(e.src)) {
              e.src = id_parser_.get_local_id(e.src);
              e.dst = parseOrAddOuterVertexGid(e.dst);
            } else {
              e.src = invalid_vid;
            }
          }
        }
      } else if (load_strategy == LoadStrategy::kOnlyOut) {
        for (auto& e : edges_to_add) {
          if (IsInnerVertexGid(e.src)) {
            e.src = id_parser_.get_local_id(e.src);
            if (!IsInnerVertexGid(e.dst)) {
              e.dst = parseOrAddOuterVertexGid(e.dst);
            } else {
              e.dst = id_parser_.get_local_id(e.dst);
            }
          } else {
            if (!directed_ && IsInnerVertexGid(e.dst)) {
              e.dst = id_parser_.get_local_id(e.dst);
              e.src = parseOrAddOuterVertexGid(e.src);
            } else {
              e.src = invalid_vid;
            }
          }
        }
      } else if (load_strategy == LoadStrategy::kBothOutIn) {
        for (auto& e : edges_to_add) {
          if (IsInnerVertexGid(e.src)) {
            e.src = id_parser_.get_local_id(e.src);
            if (IsInnerVertexGid(e.dst)) {
              e.dst = id_parser_.get_local_id(e.dst);
            } else {
              e.dst = parseOrAddOuterVertexGid(e.dst);
            }
          } else {
            if (IsInnerVertexGid(e.dst)) {
              e.src = parseOrAddOuterVertexGid(e.src);
              e.dst = id_parser_.get_local_id(e.dst);
            } else {
              e.src = invalid_vid;
            }
          }
        }
      } else {
        LOG(FATAL) << "Invalid load strategy";
      }
      vid_t new_ivnum = vm_ptr_->GetInnerVertexSize(fid_);
      vid_t new_ovnum = ovgid_.size();
      this->inner_vertices_.SetRange(0, new_ivnum);
      this->outer_vertices_.SetRange(id_parser_.max_local_id() - new_ovnum,
                                     id_parser_.max_local_id());
      this->vertices_.SetRange(0, new_ivnum,
                               id_parser_.max_local_id() - new_ovnum,
                               id_parser_.max_local_id());
      this->ivnum_ = new_ivnum;
      if (ovnum_ != new_ovnum) {
        ovnum_ = new_ovnum;
        initOuterVerticesOfFragment();
      }
      ie_.add_vertices(new_ivnum - ivnum, new_ovnum - ovnum);
      oe_.add_vertices(new_ivnum - ivnum, new_ovnum - ovnum);
      if (this->directed_) {
        ie_.add_reversed_edges(edges_to_add);
        oe_.add_forward_edges(edges_to_add);
      } else {
        ie_.add_edges(edges_to_add);
        oe_.add_edges(edges_to_add);
      }
    }
    ivdata_.resize(this->ivnum_);
    ovdata_.resize(this->ovnum_);
    for (auto& v : mutation.vertices_to_add) {
      vid_t lid;
      if (IsInnerVertexGid(v.vid)) {
        this->InnerVertexGid2Lid(v.vid, lid);
        ivdata_[lid] = v.vdata;
      } else {
        if (this->OuterVertexGid2Lid(v.vid, lid)) {
          ovdata_[outerVertexLidToIndex(lid)] = v.vdata;
        }
      }
    }
    for (auto& v : mutation.vertices_to_update) {
      vid_t lid;
      if (IsInnerVertexGid(v.vid)) {
        this->InnerVertexGid2Lid(v.vid, lid);
        ivdata_[lid] = v.vdata;
      } else {
        if (this->OuterVertexGid2Lid(v.vid, lid)) {
          ovdata_[outerVertexLidToIndex(lid)] = v.vdata;
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
    ia << ovnum_;
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();

    if (ovnum_ > 0) {
      CHECK(io_adaptor->Write(&ovgid_[0], ovnum_ * sizeof(VID_T)));
    }

    ia << ivdata_ << ovdata_;
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
    CHECK(io_adaptor->ReadArchive(oa));
    oa >> ovnum_;

    oa.Clear();

    ovgid_.clear();
    ovgid_.resize(ovnum_);
    if (ovnum_ > 0) {
      CHECK(io_adaptor->Read(&ovgid_[0], ovnum_ * sizeof(VID_T)));
    }

    initOuterVerticesOfFragment();

    {
      ovg2i_.clear();
      VID_T ovlid = id_parser_.max_local_id();
      for (auto gid : ovgid_) {
        ovg2i_.emplace(gid, --ovlid);
      }
    }

    CHECK(io_adaptor->ReadArchive(oa));
    oa >> ivdata_ >> ovdata_;

    io_adaptor->Close();
  }

  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) override {
    base_t::PrepareToRunApp(comm_spec, conf);
    if (conf.need_split_edges_by_fragment) {
      LOG(FATAL) << "MutableEdgecutFragment cannot split edges by fragment";
    } else if (conf.need_split_edges) {
      splitEdges();
    }
  }

  using base_t::InnerVertices;
  using base_t::IsInnerVertex;
  using base_t::OuterVertices;

  inline const VDATA_T& GetData(const vertex_t& v) const override {
    return IsInnerVertex(v) ? ivdata_[v.GetValue()]
                            : ovdata_[outerVertexLidToIndex(v.GetValue())];
  }

  inline void SetData(const vertex_t& v, const VDATA_T& val) override {
    if (IsInnerVertex(v)) {
      ivdata_[v.GetValue()] = val;
    } else {
      ovdata_[outerVertexLidToIndex(v.GetValue())] = val;
    }
  }

  bool OuterVertexGid2Lid(VID_T gid, VID_T& lid) const override {
    auto iter = ovg2i_.find(gid);
    if (iter != ovg2i_.end()) {
      lid = iter->second;
      return true;
    } else {
      return false;
    }
  }

  VID_T GetOuterVertexGid(vertex_t v) const override {
    return ovgid_[outerVertexLidToIndex(v.GetValue())];
  }

  inline bool Gid2Vertex(const VID_T& gid, vertex_t& v) const override {
    fid_t fid = id_parser_.get_fragment_id(gid);
    if (fid == fid_) {
      v.SetValue(id_parser_.get_local_id(gid));
      return true;
    } else {
      auto iter = ovg2i_.find(gid);
      if (iter != ovg2i_.end()) {
        v.SetValue(iter->second);
        return true;
      } else {
        return false;
      }
    }
  }

  inline VID_T Vertex2Gid(const vertex_t& v) const override {
    if (IsInnerVertex(v)) {
      return id_parser_.generate_global_id(fid_, v.GetValue());
    } else {
      return ovgid_[outerVertexLidToIndex(v.GetValue())];
    }
  }

 public:
  using base_t::GetIncomingAdjList;
  using base_t::GetOutgoingAdjList;

  fragment_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                         fid_t dst_fid) override {
    return fragment_adj_list_t(
        get_oe_begin(v), get_oe_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

  fragment_const_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                               fid_t dst_fid) const override {
    return fragment_const_adj_list_t(
        get_oe_begin(v), get_oe_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

  fragment_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                         fid_t dst_fid) override {
    return fragment_adj_list_t(
        get_ie_begin(v), get_ie_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

  fragment_const_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                               fid_t dst_fid) const override {
    return fragment_const_adj_list_t(
        get_ie_begin(v), get_ie_end(v), [this, dst_fid](const nbr_t& nbr) {
          return this->GetFragId(nbr.get_neighbor()) == dst_fid;
        });
  }

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
    return adj_list_t(get_ie_begin(v), iespliter_[v]);
  }

  inline const_adj_list_t GetIncomingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(get_ie_begin(v), iespliter_[v]);
  }

  inline adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(iespliter_[v], get_ie_end(v));
  }

  inline const_adj_list_t GetIncomingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(iespliter_[v], get_ie_end(v));
  }

  inline adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(get_oe_begin(v), oespliter_[v]);
  }

  inline const_adj_list_t GetOutgoingInnerVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(get_oe_begin(v), oespliter_[v]);
  }

  inline adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) override {
    assert(IsInnerVertex(v));
    return adj_list_t(oespliter_[v], get_oe_end(v));
  }

  inline const_adj_list_t GetOutgoingOuterVertexAdjList(
      const vertex_t& v) const override {
    assert(IsInnerVertex(v));
    return const_adj_list_t(oespliter_[v], get_oe_end(v));
  }

 private:
  inline VID_T outerVertexLidToIndex(VID_T lid) const {
    return id_parser_.max_local_id() - lid - 1;
  }

  inline VID_T outerVertexIndexToLid(VID_T index) const {
    return id_parser_.max_local_id() - index - 1;
  }

  void splitEdges() {
    auto inner_vertices = InnerVertices();
    iespliter_.Init(inner_vertices);
    oespliter_.Init(inner_vertices);
    int inner_neighbor_count = 0;
    for (auto& v : inner_vertices) {
      inner_neighbor_count = 0;
      auto ie = GetIncomingAdjList(v);
      for (auto& e : ie) {
        if (IsInnerVertex(e.neighbor)) {
          ++inner_neighbor_count;
        }
      }
      iespliter_[v] = get_ie_begin(v) + inner_neighbor_count;

      inner_neighbor_count = 0;
      auto oe = GetOutgoingAdjList(v);
      for (auto& e : oe) {
        if (IsInnerVertex(e.neighbor)) {
          ++inner_neighbor_count;
        }
      }
      oespliter_[v] = get_oe_begin(v) + inner_neighbor_count;
    }
  }

  VID_T parseOrAddOuterVertexGid(VID_T gid) {
    auto iter = ovg2i_.find(gid);
    if (iter != ovg2i_.end()) {
      return iter->second;
    } else {
      ++ovnum_;
      VID_T lid = id_parser_.max_local_id() - ovnum_;
      ovgid_.push_back(gid);
      ovg2i_.emplace(gid, lid);
      return lid;
    }
  }

  void initOuterVerticesOfFragment() {
    outer_vertices_of_frag_.resize(fnum_);
    for (auto& vec : outer_vertices_of_frag_) {
      vec.clear();
    }
    for (VID_T i = 0; i < ovnum_; ++i) {
      fid_t fid = id_parser_.get_fragment_id(ovgid_[i]);
      outer_vertices_of_frag_[fid].push_back(
          vertex_t(outerVertexIndexToLid(i)));
    }
  }

  using base_t::ivnum_;
  VID_T ovnum_;
  using base_t::directed_;
  using base_t::fid_;
  using base_t::fnum_;
  using base_t::id_parser_;

  ska::flat_hash_map<VID_T, VID_T> ovg2i_;
  std::vector<VID_T> ovgid_;
  Array<VDATA_T, Allocator<VDATA_T>> ivdata_;
  Array<VDATA_T, Allocator<VDATA_T>> ovdata_;

  VertexArray<inner_vertices_t, nbr_t*> iespliter_, oespliter_;

  using base_t::outer_vertices_of_frag_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_MUTABLE_EDGECUT_FRAGMENT_H_
