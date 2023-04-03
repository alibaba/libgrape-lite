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

#ifndef EXAMPLES_GNN_SAMPLER_APPEND_ONLY_EDGECUT_FRAGMENT_H_
#define EXAMPLES_GNN_SAMPLER_APPEND_ONLY_EDGECUT_FRAGMENT_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <grape/config.h>
#include <grape/fragment/basic_fragment_loader.h>
#include <grape/fragment/edgecut_fragment_base.h>
#include <grape/fragment/fragment_base.h>
#include <grape/fragment/partitioner.h>
#include <grape/graph/adj_list.h>
#include <grape/graph/edge.h>
#include <grape/graph/vertex.h>
#include <grape/io/io_adaptor_base.h>
#include <grape/io/tsv_line_parser.h>
#include <grape/serialization/in_archive.h>
#include <grape/serialization/out_archive.h>
#include <grape/types.h>
#include <grape/utils/gcontainer.h>
#include <grape/utils/iterator_pair.h>
#include <grape/utils/vertex_array.h>
#include <grape/vertex_map/global_vertex_map.h>
#include <grape/worker/comm_spec.h>

#include "flat_hash_map/flat_hash_map.hpp"
#include "fragment_indices.h"

namespace grape {

/** InsertOnly Edgecut Fragment
 *
 * It is a kind of strategy to load fragment by a way of insert only.
 * Fragments can be extended with extra vertices and edges in this way. All
 * data needed for the running of an App are stored in here following its
 * strategy.
 */

template <typename VID_T, typename EDATA_T>
class NbrIterImpl {
 public:
  NbrIterImpl() {}
  virtual ~NbrIterImpl() {}

  virtual NbrIterImpl* clone() = 0;

  virtual void Inc() = 0;
  virtual void Add(int right) = 0;

  virtual const Nbr<VID_T, EDATA_T>& deref() = 0;
  virtual const Nbr<VID_T, EDATA_T>* pointer() = 0;
};

template <typename VID_T, typename EDATA_T>
class NbrVertex {
 public:
  Nbr<VID_T, EDATA_T> vertex;
  NbrVertex* Next() { return next_ ? (this + next_) : nullptr; }
  void SetNext(NbrVertex* next) { next_ = next - this; }

 private:
  int next_;
};

template <typename VID_T, typename EDATA_T>
class NbrIterator {
 public:
  NbrIterator() : impl_(NULL) {}
  explicit NbrIterator(NbrIterImpl<VID_T, EDATA_T>* impl) : impl_(impl) {}
  NbrIterator(NbrIterator const& right) : impl_(right.impl_->clone()) {}

  ~NbrIterator() { delete impl_; }

  NbrIterator& operator=(NbrIterator const& right) {
    delete impl_;
    impl_ = right.impl_->clone();
    return *this;
  }

  NbrIterator& operator+=(int right) {
    impl_->Add(right);
    return *this;
  }

  NbrIterator& operator++() {
    impl_->Inc();
    return *this;
  }

  const NbrIterator operator++(int) {
    NbrIterator it(*this);
    impl_->Inc();
    return it;
  }

  Nbr<VID_T, EDATA_T>& operator*() { return impl_->deref(); }
  Nbr<VID_T, EDATA_T>* operator->() { return impl_->pointer(); }

  Nbr<VID_T, EDATA_T>& nbr() { return impl_->deref(); }
  Nbr<VID_T, EDATA_T>* nbr_pointer() { return impl_->pointer(); }

  bool equals(const NbrIterator& it) {
    return (impl_->pointer() == it.impl_->pointer());
  }

  bool equals(const NbrIterator* it) { return this->equals(*it); }

  bool operator==(const NbrIterator& it) { return this->equals(it); }
  bool operator!=(const NbrIterator& it) { return !this->equals(it); }

 private:
  NbrIterImpl<VID_T, EDATA_T>* impl_;
};

template <typename VID_T, typename EDATA_T>
class NbrSpaceIterImpl : public NbrIterImpl<VID_T, EDATA_T> {
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using nbr_vertex_t = NbrVertex<VID_T, EDATA_T>;

 public:
  NbrSpaceIterImpl() : linked_list_ptr_(NULL) {}

  explicit NbrSpaceIterImpl(nbr_vertex_t* linked_list_ptr)
      : linked_list_ptr_(linked_list_ptr) {}

  NbrSpaceIterImpl* clone() { return new NbrSpaceIterImpl(linked_list_ptr_); }

  void Inc() {
    if (linked_list_ptr_) {
      linked_list_ptr_ = linked_list_ptr_->Next();
    }
  }

  void Add(int right) {
    for (int i = 0; i < right; ++i) {
      Inc();
    }
  }

  nbr_t& deref() {
    // may trigger an NPE (segmentation fault) as error
    return linked_list_ptr_->vertex;
  }

  nbr_t* pointer() {
    if (linked_list_ptr_) {
      return &linked_list_ptr_->vertex;
    }
    return nullptr;  // return null pointer
  }

 private:
  nbr_vertex_t* linked_list_ptr_;
};

template <typename VID_T, typename EDATA_T>
class NbrMapSpaceIterImpl : public NbrIterImpl<VID_T, EDATA_T> {
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using nbr_vertex_t = NbrVertex<VID_T, EDATA_T>;

 public:
  NbrMapSpaceIterImpl() {}

  explicit NbrMapSpaceIterImpl(
      typename std::map<VID_T, Nbr<VID_T, EDATA_T>>::const_iterator map_iter)
      : map_iter_(map_iter) {}

  inline NbrMapSpaceIterImpl* clone() {
    return new NbrMapSpaceIterImpl(map_iter_);
  }

  inline void Inc() { map_iter_++; }

  inline void Add(int right) {
    for (int i = 0; i < right; ++i) {
      Inc();
    }
  }

  inline const nbr_t& deref() {
    // may trigger an NPE (segmentation fault) as error
    return map_iter_->second;
  }

  inline const nbr_t* pointer() {
    // may trigger an NPE (segmentation fault) as error
    return &map_iter_->second;
  }

 private:
  typename std::map<VID_T, Nbr<VID_T, EDATA_T>>::const_iterator map_iter_;
};

template <typename VID_T, typename EDATA_T>
class NbrMapSpace {
 public:
  NbrMapSpace() : index_(0) {}

  // Create a new linked list
  inline size_t emplace(VID_T vid, EDATA_T edata) {
    buffer_.resize(index_ + 1);
    buffer_[index_] = new std::map<VID_T, Nbr<VID_T, EDATA_T>>();
    buffer_[index_]->operator[](vid) = Nbr<VID_T, EDATA_T>(vid, edata);
    return index_++;
  }

  // Insert the value to an existing linked list, or update the existing value
  inline size_t emplace(size_t loc, VID_T vid, EDATA_T edata, bool& created) {
    if (buffer_[loc]->find(vid) != buffer_[loc]->end()) {
      buffer_[loc]->operator[](vid) = Nbr<VID_T, EDATA_T>(vid, edata);
      created = false;
      return loc;
    } else {
      buffer_[loc]->operator[](vid) = Nbr<VID_T, EDATA_T>(vid, edata);
      created = true;
      return loc;
    }
  }

  inline void update(size_t loc, VID_T vid, EDATA_T edata) {
    if (buffer_[loc]->find(vid) != buffer_[loc]->end()) {
      buffer_[loc]->operator[](vid) = Nbr<VID_T, EDATA_T>(vid, edata);
    }
  }

  inline std::map<VID_T, Nbr<VID_T, EDATA_T>>& operator[](size_t loc) {
    return *buffer_[loc];
  }

  inline const std::map<VID_T, Nbr<VID_T, EDATA_T>>& operator[](
      size_t loc) const {
    return *buffer_[loc];
  }

  void Clear() {
    for (size_t i = 0; i < buffer_.size(); ++i) {
      delete buffer_[i];
      buffer_[i] = nullptr;
    }
    buffer_.clear();
    index_ = 0;
  }

 private:
  std::vector<std::map<VID_T, Nbr<VID_T, EDATA_T>>*> buffer_;
  size_t index_;
};

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
struct AppendOnlyEdgecutFragmentTraits {
  using inner_vertices_t = VertexRange<VID_T>;
  using outer_vertices_t = VertexRange<VID_T>;
  using vertices_t = DualVertexRange<VID_T>;

  using sub_vertices_t = VertexVector<VID_T>;
  using fragment_adj_list_t = AdjList<VID_T, EDATA_T>;
  using fragment_const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using vertex_map_t = GlobalVertexMap<OID_T, VID_T>;
  using mirror_vertices_t = std::vector<Vertex<VID_T>>;
};

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy _load_strategy = LoadStrategy::kBothOutIn>
class AppendOnlyEdgecutFragment
    : public EdgecutFragmentBase<
          OID_T, VID_T, VDATA_T, EDATA_T,
          AppendOnlyEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T>> {
 public:
  using traits_t =
      AppendOnlyEdgecutFragmentTraits<OID_T, VID_T, VDATA_T, EDATA_T>;
  using base_t = EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>;
  using internal_vertex_t = internal::Vertex<VID_T, VDATA_T>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using nbr_iterator_t = NbrIterator<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_map_t = typename traits_t::vertex_map_t;
  using nbr_space_iter_impl = NbrSpaceIterImpl<VID_T, EDATA_T>;
  using nbr_mapspace_iter_impl = NbrMapSpaceIterImpl<VID_T, EDATA_T>;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  using inner_vertices_t = typename traits_t::inner_vertices_t;
  using outer_vertices_t = typename traits_t::outer_vertices_t;
  using vertices_t = typename traits_t::vertices_t;

  template <typename T>
  using inner_vertex_array_t = VertexArray<inner_vertices_t, T>;

  template <typename T>
  using outer_vertex_array_t = VertexArray<outer_vertices_t, T>;

  template <typename T>
  using vertex_array_t = VertexArray<vertices_t, T>;

  /** Constructor.
   * @param vm_ptr the vertex map.
   */
  explicit AppendOnlyEdgecutFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>(vm_ptr) {}

  virtual ~AppendOnlyEdgecutFragment() {}

  using base_t::Gid2Lid;
  using base_t::init;
  using base_t::InnerVertexGid2Lid;
  using base_t::IsInnerVertexGid;
  static std::string type_info() { return ""; }
  void Init(fid_t fid, bool directed, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) override {
    init(fid, directed);

    ovnum_ = 0;
    oenum_ = 0;
    oe_.clear();
    oeoffset_.clear();

    ovg2i_.clear();

    vid_t invalid_vid = std::numeric_limits<vid_t>::max();
    {
      std::vector<vid_t> outer_vertices;
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.src)) {
          if (!IsInnerVertexGid(e.dst)) {
            outer_vertices.push_back(e.dst);
          }
        } else {
          if (!directed && IsInnerVertexGid(e.dst)) {
            outer_vertices.push_back(e.src);
          } else {
            e.src = invalid_vid;
          }
        }
      }
      DistinctSort(outer_vertices);
      ovgid_.resize(outer_vertices.size());
      memcpy(&ovgid_[0], &outer_vertices[0],
             outer_vertices.size() * sizeof(vid_t));
    }

    ovnum_ = 0;
    for (auto gid : ovgid_) {
      ovg2i_.emplace(gid, ovnum_);
      ++ovnum_;
    }
    tvnum_ = ivnum_ + ovnum_;
    max_old_ilid_ = ivnum_;
    min_old_olid_ = id_parser_.max_local_id() - ovnum_;
    this->inner_vertices_.SetRange(0, ivnum_);
    this->outer_vertices_.SetRange(id_parser_.max_local_id() - ovnum_,
                                   id_parser_.max_local_id());
    this->vertices_.SetRange(0, ivnum_, id_parser_.max_local_id() - ovnum_,
                             id_parser_.max_local_id());

    {
      std::vector<int> odegree(ivnum_, 0);
      oenum_ = 0;
      if (directed) {
        for (auto& e : edges) {
          if (e.src != invalid_vid) {
            InnerVertexGid2Lid(e.src, e.src);
            Gid2Lid(e.dst, e.dst);
            ++odegree[e.src];
            ++oenum_;
          }
        }
      } else {
        for (auto& e : edges) {
          if (e.src != invalid_vid) {
            if (IsInnerVertexGid(e.src)) {
              InnerVertexGid2Lid(e.src, e.src);
              if (IsInnerVertexGid(e.dst)) {
                InnerVertexGid2Lid(e.dst, e.dst);
                ++odegree[e.dst];
                ++oenum_;
              } else {
                OuterVertexGid2Lid(e.dst, e.dst);
              }
              ++odegree[e.src];
              ++oenum_;
            } else {
              InnerVertexGid2Lid(e.dst, e.dst);
              OuterVertexGid2Lid(e.src, e.src);
              ++odegree[e.dst];
              ++oenum_;
            }
          }
        }
      }
      oe_.resize(oenum_);
      oeoffset_.resize(ivnum_ + 1);
      oeoffset_[0] = &oe_[0];
      for (vid_t i = 0; i < ivnum_; ++i) {
        oeoffset_[i + 1] = oeoffset_[i] + odegree[i];
      }
    }

    {
      Array<nbr_t*, Allocator<nbr_t*>> oeiter(oeoffset_);
      if (directed) {
        for (auto& e : edges) {
          if (e.src != invalid_vid) {
            oeiter[e.src]->neighbor = e.dst;
            oeiter[e.src]->data = e.edata;
            ++oeiter[e.src];
          }
        }
      } else {
        for (auto& e : edges) {
          if (e.src != invalid_vid) {
            if (e.src < ivnum_) {
              oeiter[e.src]->neighbor = e.dst;
              oeiter[e.src]->data = e.edata;
              ++oeiter[e.src];
              if (e.dst < ivnum_) {
                oeiter[e.dst]->neighbor = e.src;
                oeiter[e.dst]->data = e.edata;
                ++oeiter[e.dst];
              }
            } else {
              oeiter[e.dst]->neighbor = e.src;
              oeiter[e.dst]->data = e.edata;
              ++oeiter[e.dst];
            }
          }
        }
      }
    }

    for (vid_t i = 0; i < ivnum_; ++i) {
      std::sort(oeoffset_[i], oeoffset_[i + 1],
                [](const nbr_t& lhs, const nbr_t& rhs) {
                  return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                });
    }

    extra_oenum_ = 0;
    extra_oe_.clear();
    extra_oe_.resize(ivnum_, -1);

    ivdata_.clear();
    ivdata_.resize(ivnum_);
    if (sizeof(internal_vertex_t) > sizeof(vid_t)) {
      for (auto& v : vertices) {
        vid_t gid = v.vid;
        if (id_parser_.get_fragment_id(gid) == fid_) {
          ivdata_[id_parser_.get_local_id(gid)] = v.vdata;
        }
      }
    }

    initOuterVerticesOfFragment();
  }

  void InitIndices() {
    fragment_indices_ =
        FragmentIndicesBase<oid_t, vid_t, vdata_t, edata_t>::Create();
    fragment_indices_->Init(this);
  }

  void ExtendFragment(std::vector<std::string>& edge_messages,
                      const CommSpec& comm_spec, const LoadGraphSpec& spec) {
    sync_comm::Bcast(edge_messages, kCoordinatorRank, comm_spec.comm());

    if (edge_messages.empty()) {
      return;
    }

    std::vector<edge_t> edges;
    edges.reserve(edge_messages.size());
    std::vector<oid_t> empty_id_list;
    auto& partitioner = vm_ptr_->GetPartitioner();
    {
      edata_t e_data;
      oid_t src, dst, src_gid, dst_gid;
      fid_t src_fid, dst_fid;
      auto line_parser_ptr =
          std::make_shared<TSVLineParser<oid_t, vdata_t, edata_t>>();
      for (auto& msg : edge_messages) {
        if (msg.empty() || msg[0] == '#') {
          continue;
        }
        try {
          line_parser_ptr->LineParserForEFile(msg, src, dst, e_data);
        } catch (std::exception& e) {
          LOG(ERROR) << e.what();
          continue;
        }
        src_fid = partitioner.GetPartitionId(src);
        dst_fid = partitioner.GetPartitionId(dst);
        vm_ptr_->AddVertex(src, src_gid);
        vm_ptr_->AddVertex(dst, dst_gid);
        if (src_fid == fid_ || dst_fid == fid_) {
          edges.emplace_back(src_gid, dst_gid, e_data);
        }
      }
    }

    ivnum_ = vm_ptr_->GetInnerVertexSize(fid_);
    extra_oe_.resize(ivnum_, -1);
    fragment_indices_->Resize(ivnum_);

    {
      vid_t src_lid, dst_lid;
      vid_t old_ovnum = ovnum_;
      std::vector<vid_t> ov_to_extend;
      for (auto& e : edges) {
        if (IsInnerVertexGid(e.src)) {
          src_lid = id_parser_.get_local_id(e.src);
          if (IsInnerVertexGid(e.dst)) {
            dst_lid = id_parser_.get_local_id(e.dst);
            if (!spec.directed) {
              addOutgoingEdge(dst_lid, src_lid, e.edata);
              fragment_indices_->Insert(dst_lid, e.src, e.edata);
            }
          } else {
            auto iter = ovg2i_.find(e.dst);
            if (iter != ovg2i_.end()) {
              dst_lid = id_parser_.max_local_id() - iter->second;
            } else {
              ovg2i_.emplace(e.dst, ovnum_);
              ov_to_extend.emplace_back(e.dst);
              dst_lid = id_parser_.max_local_id() - ovnum_;
              ++ovnum_;
            }
          }
          addOutgoingEdge(src_lid, dst_lid, e.edata);
          fragment_indices_->Insert(src_lid, e.dst, e.edata);
        } else if (!spec.directed && IsInnerVertexGid(e.dst)) {
          dst_lid = id_parser_.get_local_id(e.dst);
          auto iter = ovg2i_.find(e.src);
          if (iter != ovg2i_.end()) {
            src_lid = id_parser_.max_local_id() - iter->second;
          } else {
            ovg2i_.emplace(e.src, ovnum_);
            ov_to_extend.emplace_back(e.src);
            src_lid = id_parser_.max_local_id() - ovnum_;
            ++ovnum_;
          }
          addOutgoingEdge(dst_lid, src_lid, e.edata);
          fragment_indices_->Insert(dst_lid, e.src, e.edata);
        }
      }
      this->inner_vertices_.SetRange(0, ivnum_);
      this->outer_vertices_.SetRange(id_parser_.max_local_id() - ovnum_,
                                     id_parser_.max_local_id());
      this->vertices_.SetRange(0, ivnum_, id_parser_.max_local_id() - ovnum_,
                               id_parser_.max_local_id());
      tvnum_ = ivnum_ + ovnum_;
      ovgid_.resize(ovnum_);
      memcpy(&ovgid_[old_ovnum], &ov_to_extend[0],
             sizeof(vid_t) * ov_to_extend.size());
      if (old_ovnum != ovnum_) {
        initOuterVerticesOfFragment();
      }
    }

    fragment_indices_->Rebuild();
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid_);
    VLOG(1) << "Serialize to " << fbuf;

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    base_t::serialize(io_adaptor);

    InArchive ia;

    vid_t xivnum = max_old_ilid_;
    vid_t xovnum = id_parser_.max_local_id() - min_old_olid_;

    ia << xivnum << xovnum << oenum_;
    io_adaptor->WriteArchive(ia);
    ia.Clear();

    std::map<vid_t, vid_t> xovg2i;
    for (auto& pair : ovg2i_) {
      if (pair.second < xovnum) {
        xovg2i[pair.first] = pair.second;
      }
    }

    ia << xovg2i;
    io_adaptor->WriteArchive(ia);
    ia.Clear();

    io_adaptor->Write(&ovgid_[0], xovnum * sizeof(vid_t));

    {
      if (std::is_pod<EDATA_T>::value || (sizeof(nbr_t) == sizeof(VID_T))) {
        if (oenum_ > 0) {
          CHECK(io_adaptor->Write(&oe_[0], oenum_ * sizeof(nbr_t)));
        }
      } else {
        ia << oe_;
        CHECK(io_adaptor->WriteArchive(ia));
        ia.Clear();
      }

      std::vector<int> odegree(xivnum);
      for (VID_T i = 0; i < xivnum; ++i) {
        odegree[i] = oeoffset_[i + 1] - oeoffset_[i];
      }
      CHECK(io_adaptor->Write(&odegree[0], sizeof(int) * xivnum));
    }

    io_adaptor->Close();
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string prefix, const fid_t fid) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid);
    VLOG(1) << "Deserialize from " << fbuf;
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    OutArchive oa;
    CHECK(io_adaptor->ReadArchive(oa));
    oa >> ivnum_ >> ovnum_ >> oenum_;
    tvnum_ = ivnum_ + ovnum_;
    oa.Clear();

    CHECK(io_adaptor->ReadArchive(oa));
    oa >> ovg2i_;
    oa.Clear();

    ovgid_.clear();
    ovgid_.resize(ovnum_);
    CHECK(io_adaptor->Read(&ovgid_[0], ovnum_ * sizeof(vid_t)));

    oe_.clear();
    if (std::is_pod<EDATA_T>::value || (sizeof(nbr_t) == sizeof(vid_t))) {
      oe_.resize(oenum_);
      if (oenum_ > 0) {
        CHECK(io_adaptor->Read(&oe_[0], oenum_ * sizeof(nbr_t)));
      }
    } else {
      CHECK(io_adaptor->ReadArchive(oa));
      oa >> oe_;
      oa.Clear();
      CHECK_EQ(oe_.size(), oenum_);
    }

    oeoffset_.clear();
    oeoffset_.clear();
    oeoffset_.resize(ivnum_ + 1);
    oeoffset_[0] = &oe_[0];
    {
      std::vector<int> odegree(tvnum_);
      CHECK(io_adaptor->Read(&odegree[0], sizeof(int) * tvnum_));
      for (VID_T i = 0; i < ivnum_; ++i) {
        oeoffset_[i + 1] = oeoffset_[i] + odegree[i];
      }
    }

    ivdata_.clear();
    ivdata_.resize(ivnum_);

    io_adaptor->Close();

    max_old_ilid_ = ivnum_;
    min_old_olid_ = id_parser_.max_local_id() - ovnum_;
    extra_oenum_ = 0;
    extra_oe_.clear();
    extra_oe_.resize(ivnum_, -1);
    this->inner_vertices_.SetRange(0, ivnum_);
    this->outer_vertices_.SetRange(id_parser_.max_local_id() - ovnum_,
                                   id_parser_.max_local_id());
    this->vertices_.SetRange(0, ivnum_, id_parser_.max_local_id() - ovnum_,
                             id_parser_.max_local_id());

    initOuterVerticesOfFragment();
  }

  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) override {}

  fid_t GetFragIdByGid(const vid_t& gid) const {
    return id_parser_.get_fragment_id(gid);
  }

  size_t GetEdgeNum() const override { return oenum_ + extra_oenum_; }

  const vdata_t& GetData(const vertex_t& v) const override {
    return ivdata_[v.GetValue()];
  }

  using base_t::IsInnerVertex;

  void SetData(const vertex_t& v, const vdata_t& val) override {
    if (IsInnerVertex(v)) {
      ivdata_[v.GetValue()] = val;
    }
  }

  bool HasChild(const vertex_t& v) const override {
    auto extra_oes = GetExtraOutgoingAdjList(v);
    return GetOutgoingAdjList(v).NotEmpty() ||
           (extra_oes.begin() != extra_oes.end());
  }

  bool HasParent(const vertex_t& v) const override { return false; }

  int GetLocalOutDegree(const vertex_t& v) const override {
    int od = GetOutgoingAdjList(v).Size();
    auto es = GetExtraOutgoingAdjList(v);
    for (auto e = es.begin(); e != es.end(); ++e) {
      od += 1;
    }
    return od;
  }

  int GetLocalInDegree(const vertex_t& v) const override { return 0; }

  bool OuterVertexGid2Lid(VID_T gid, VID_T& lid) const override {
    auto iter = ovg2i_.find(gid);
    if (iter != ovg2i_.end()) {
      lid = id_parser_.max_local_id() - iter->second;
      return true;
    } else {
      return false;
    }
  }

  VID_T GetOuterVertexGid(vertex_t v) const override {
    return ovgid_[id_parser_.max_local_id() - v.GetValue()];
  }

  // AppendOnlyEdgecutFragment doesn't support along edge message strategy
  DestList IEDests(const vertex_t& v) const override {
    LOG(FATAL) << "Not implemented.";
    return DestList(0, 0);
  }

  DestList OEDests(const vertex_t& v) const override {
    LOG(FATAL) << "Not implemented.";
    return DestList(0, 0);
  }

  DestList IOEDests(const vertex_t& v) const override {
    LOG(FATAL) << "Not implemented.";
    return DestList(0, 0);
  }

  adj_list_t GetIncomingAdjList(const vertex_t& v) override {
    LOG(FATAL) << "Not implemented.";
    return adj_list_t(NULL, NULL);
  }

  const_adj_list_t GetIncomingAdjList(const vertex_t& v) const override {
    LOG(FATAL) << "Not implemented.";
    return const_adj_list_t(NULL, NULL);
  }

  adj_list_t GetOutgoingAdjList(const vertex_t& v) override {
    return (v.GetValue() < max_old_ilid_)
               ? adj_list_t(oeoffset_[v.GetValue()],
                            oeoffset_[v.GetValue() + 1])
               : adj_list_t(NULL, NULL);
  }

  const_adj_list_t GetOutgoingAdjList(const vertex_t& v) const override {
    return (v.GetValue() < max_old_ilid_)
               ? const_adj_list_t(oeoffset_[v.GetValue()],
                                  oeoffset_[v.GetValue() + 1])
               : const_adj_list_t(NULL, NULL);
  }

  inline adj_list_t GetIncomingAdjList(const vertex_t& v,
                                       fid_t src_fid) override {
    return adj_list_t(NULL, NULL);
  }

  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v,
                                             fid_t src_fid) const override {
    return const_adj_list_t(NULL, NULL);
  }

  inline adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                       fid_t dst_fid) override {
    return adj_list_t(NULL, NULL);
  }

  inline const_adj_list_t GetOutgoingAdjList(const vertex_t& v,
                                             fid_t dst_fid) const override {
    return const_adj_list_t(NULL, NULL);
  }

  adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) override {
    LOG(FATAL) << "Not implemented.";
    return adj_list_t(NULL, NULL);
  }

  const_adj_list_t GetIncomingInnerVertexAdjList(
      const vertex_t& v) const override {
    LOG(FATAL) << "Not implemented.";
    return const_adj_list_t(NULL, NULL);
  }

  adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) override {
    return (v.GetValue() < max_old_ilid_)
               ? adj_list_t(oeoffset_[v.GetValue()],
                            oeoffset_[v.GetValue() + 1])
               : adj_list_t(NULL, NULL);
  }

  const_adj_list_t GetOutgoingInnerVertexAdjList(
      const vertex_t& v) const override {
    return (v.GetValue() < max_old_ilid_)
               ? const_adj_list_t(oeoffset_[v.GetValue()],
                                  oeoffset_[v.GetValue() + 1])
               : const_adj_list_t(NULL, NULL);
  }

  adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) override {
    LOG(FATAL) << "Not implemented.";
    return adj_list_t(NULL, NULL);
  }

  const_adj_list_t GetIncomingOuterVertexAdjList(
      const vertex_t& v) const override {
    LOG(FATAL) << "Not implemented.";
    return const_adj_list_t(NULL, NULL);
  }

  adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) override {
    return adj_list_t(NULL, NULL);
  }

  const_adj_list_t GetOutgoingOuterVertexAdjList(
      const vertex_t& v) const override {
    return const_adj_list_t(NULL, NULL);
  }

  size_t GetOriginalEdgeNum() const { return oenum_; }

  size_t GetExtraEdgeNum() const { return extra_oenum_; }

  IteratorPair<nbr_iterator_t> GetExtraOutgoingAdjList(
      const vertex_t& v) const {
    int loc = extra_oe_[v.GetValue()];

    if (loc >= 0) {
      return IteratorPair<nbr_iterator_t>(
          nbr_iterator_t(
              new nbr_mapspace_iter_impl(extra_edge_space_[loc].begin())),
          nbr_iterator_t(
              new nbr_mapspace_iter_impl(extra_edge_space_[loc].end())));
    } else {
      return IteratorPair<nbr_iterator_t>(
          nbr_iterator_t(new nbr_space_iter_impl(nullptr)),
          nbr_iterator_t(new nbr_space_iter_impl(nullptr)));
    }
  }

  IteratorPair<nbr_iterator_t> GetExtraIncomingAdjList(
      const vertex_t& v) const {
    LOG(FATAL) << "Not implemented.";
    return IteratorPair<nbr_iterator_t>(
        nbr_iterator_t(new nbr_space_iter_impl(nullptr)),
        nbr_iterator_t(new nbr_space_iter_impl(nullptr)));
  }

  const std::unique_ptr<FragmentIndicesBase<oid_t, vid_t, vdata_t, edata_t>>&
  GetFragmentIndices() const {
    return fragment_indices_;
  }

 private:
  bool binarySearchAndUpdate(adj_list_t& edge_list, vid_t lid,
                             const edata_t& edata) {
    if (edge_list.NotEmpty()) {
      nbr_t* l = edge_list.begin_pointer();
      nbr_t* r = edge_list.end_pointer() - 1;
      while (l <= r) {
        nbr_t* m = l + (r - l) / 2;
        if (m->neighbor.GetValue() == lid) {
          m->data = edata;
          return true;
        } else if (m->neighbor.GetValue() < lid) {
          l = m + 1;
        } else {
          r = m - 1;
        }
      }
    }
    return false;
  }

  bool addOutgoingEdge(vid_t u_lid, vid_t v_lid, const edata_t& edata) {
    auto oes = GetOutgoingAdjList(vertex_t(u_lid));
    if (binarySearchAndUpdate(oes, v_lid, edata)) {
      return false;
    }
    int loc = extra_oe_[u_lid];
    if (loc == -1) {
      ++extra_oenum_;
      extra_oe_[u_lid] = extra_edge_space_.emplace(v_lid, edata);
      return true;
    } else {
      bool created = false;
      extra_oe_[u_lid] = extra_edge_space_.emplace(loc, v_lid, edata, created);
      if (created) {
        ++extra_oenum_;
      }
      return created;
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
          vertex_t(id_parser_.max_local_id() - i));
    }
  }

  vid_t ovnum_, tvnum_;
  vid_t max_old_ilid_, min_old_olid_;
  size_t oenum_;
  size_t extra_oenum_;
  using base_t::directed_;
  using base_t::fid_;
  using base_t::fnum_;
  using base_t::id_parser_;
  using base_t::ivnum_;
  using base_t::vm_ptr_;

  ska::flat_hash_map<vid_t, vid_t> ovg2i_;
  std::vector<vid_t> ovgid_;
  std::vector<vdata_t> ivdata_;
  Array<nbr_t, Allocator<nbr_t>> oe_;
  Array<nbr_t*, Allocator<nbr_t*>> oeoffset_;
  Array<int, Allocator<int>> extra_oe_;
  NbrMapSpace<vid_t, edata_t> extra_edge_space_;

  std::unique_ptr<FragmentIndicesBase<oid_t, vid_t, vdata_t, edata_t>>
      fragment_indices_;

  using base_t::outer_vertices_of_frag_;
};

}  // namespace grape
#endif  // EXAMPLES_GNN_SAMPLER_APPEND_ONLY_EDGECUT_FRAGMENT_H_
