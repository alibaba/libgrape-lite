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
#include <grape/fragment/edgecut_fragment_base.h>
#include <grape/fragment/ev_fragment_loader.h>
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

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          LoadStrategy _load_strategy = LoadStrategy::kBothOutIn>
class AppendOnlyEdgecutFragment
    : public EdgecutFragmentBase<OID_T, VID_T, VDATA_T, EDATA_T> {
 public:
  using internal_vertex_t = internal::Vertex<VID_T, VDATA_T>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using nbr_vertex_t = NbrVertex<VID_T, EDATA_T>;
  using nbr_iterator_t = NbrIterator<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using vertex_range_t = VertexRange<VID_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  template <typename DATA_T>
  using vertex_array_t = VertexArray<DATA_T, vid_t>;
  using nbr_space_iter_impl = NbrSpaceIterImpl<VID_T, EDATA_T>;
  using nbr_mapspace_iter_impl = NbrMapSpaceIterImpl<VID_T, EDATA_T>;

  using vertex_map_t = GlobalVertexMap<oid_t, vid_t>;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  AppendOnlyEdgecutFragment() {}

  /** Constructor.
   * @param vm_ptr the vertex map.
   */
  explicit AppendOnlyEdgecutFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : vm_ptr_(vm_ptr) {}

  virtual ~AppendOnlyEdgecutFragment() {}

  void Init(fid_t fid, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) override {
    fid_ = fid;
    fnum_ = vm_ptr_->GetFragmentNum();
    calcFidBitWidth(fnum_, id_mask_, fid_offset_);

    ivnum_ = vm_ptr_->GetInnerVertexSize(fid);
    ovnum_ = 0;

    oenum_ = 0;
    oe_.clear();
    oeoffset_.clear();

    ovg2i_.clear();

    vid_t invalid_vid = std::numeric_limits<vid_t>::max();
    auto is_iv_gid = [this](vid_t id) { return (id >> fid_offset_) == fid_; };
    {
      std::vector<vid_t> outer_vertices;
      for (auto& e : edges) {
        if (is_iv_gid(e.src())) {
          if (!is_iv_gid(e.dst())) {
            outer_vertices.push_back(e.dst());
          }
        } else {
          e.set_src(invalid_vid);
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
    min_old_olid_ = id_mask_ - ovnum_;

    {
      std::vector<int> odegree(ivnum_, 0);
      oenum_ = 0;
      auto gid_to_lid = [this](vid_t gid) {
        return ((gid >> fid_offset_) == fid_) ? (gid & id_mask_)
                                              : (id_mask_ - ovg2i_.at(gid));
      };
      auto iv_gid_to_lid = [this](vid_t gid) { return gid & id_mask_; };
      for (auto& e : edges) {
        if (e.src() != invalid_vid) {
          e.set_src(iv_gid_to_lid(e.src()));
          e.set_dst(gid_to_lid(e.dst()));
          ++odegree[e.src()];
          ++oenum_;
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
      for (auto& e : edges) {
        if (e.src() != invalid_vid) {
          oeiter[e.src()]->GetEdgeDst(e);
          ++oeiter[e.src()];
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
        vid_t gid = v.vid();
        if (gid >> fid_offset_ == fid_) {
          ivdata_[(gid & id_mask_)] = v.vdata();
        }
      }
    }
  }

  void InitIndices() {
    fragment_indices_ =
        FragmentIndicesBase<oid_t, vid_t, vdata_t, edata_t>::Create();
    fragment_indices_->Init(this);
  }

  void ExtendFragment(std::vector<std::string>& edge_messages,
                      const CommSpec& comm_spec, const LoadGraphSpec& spec) {
    if (comm_spec.worker_id() == kCoordinatorRank) {
      BcastSend(edge_messages, comm_spec.comm());
    } else {
      BcastRecv(edge_messages, comm_spec.comm(), kCoordinatorRank);
    }

    if (edge_messages.empty()) {
      return;
    }

    std::vector<edge_t> edges;
    edges.reserve(edge_messages.size());
    std::vector<oid_t> empty_id_list;
    HashPartitioner<oid_t> partitioner(fnum_, empty_id_list);
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
        vm_ptr_->AddVertex(src_fid, src, src_gid);
        vm_ptr_->AddVertex(dst_fid, dst, dst_gid);
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
      auto is_iv_gid = [this](vid_t id) { return (id >> fid_offset_) == fid_; };
      for (auto& e : edges) {
        if (is_iv_gid(e.src())) {
          src_lid = e.src() & id_mask_;
          if (is_iv_gid(e.dst())) {
            dst_lid = e.dst() & id_mask_;
            if (!spec.directed) {
              addOutgoingEdge(dst_lid, src_lid, e.edata());
              fragment_indices_->Insert(dst_lid, e.src(), e.edata());
            }
          } else {
            auto iter = ovg2i_.find(e.dst());
            if (iter != ovg2i_.end()) {
              dst_lid = id_mask_ - iter->second;
            } else {
              ovg2i_.emplace(e.dst(), ovnum_);
              ov_to_extend.emplace_back(e.dst());
              dst_lid = id_mask_ - ovnum_;
              ++ovnum_;
            }
          }
          addOutgoingEdge(src_lid, dst_lid, e.edata());
          fragment_indices_->Insert(src_lid, e.dst(), e.edata());
        } else if (!spec.directed && is_iv_gid(e.dst())) {
          dst_lid = e.dst() & id_mask_;
          auto iter = ovg2i_.find(e.src());
          if (iter != ovg2i_.end()) {
            src_lid = id_mask_ - iter->second;
          } else {
            ovg2i_.emplace(e.src(), ovnum_);
            ov_to_extend.emplace_back(e.src());
            src_lid = id_mask_ - ovnum_;
            ++ovnum_;
          }
          addOutgoingEdge(dst_lid, src_lid, e.edata());
          fragment_indices_->Insert(dst_lid, e.src(), e.edata());
        }
      }
      tvnum_ = ivnum_ + ovnum_;
      ovgid_.resize(ovnum_);
      memcpy(&ovgid_[old_ovnum], &ov_to_extend[0],
             sizeof(vid_t) * ov_to_extend.size());
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
    InArchive ia;

    vid_t xivnum = max_old_ilid_;
    vid_t xovnum = id_mask_ - min_old_olid_;

    ia << xivnum << xovnum << oenum_ << fid_offset_ << fid_ << id_mask_
       << fnum_;
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
    oa >> ivnum_ >> ovnum_ >> oenum_ >> fid_offset_ >> fid_ >> id_mask_ >>
        fnum_;
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
    min_old_olid_ = id_mask_ - ovnum_;
    extra_oenum_ = 0;
    extra_oe_.clear();
    extra_oe_.resize(ivnum_, -1);
  }

  void PrepareToRunApp(MessageStrategy strategy,
                       bool need_split_edge) override {}

  fid_t fid() const override { return fid_; }

  fid_t fnum() const override { return fnum_; }

  vid_t id_mask() const { return id_mask_; }

  int fid_offfset() const { return fid_offset_; }

  fid_t GetFragId(const vertex_t& u) const override {
    return IsInnerVertex(u)
               ? fid_
               : (fid_t)(ovgid_[id_mask_ - u.GetValue()] >> fid_offset_);
  }

  fid_t GetFragIdByGid(const vid_t& gid) const {
    return (fid_t)(gid >> fid_offset_);
  }

  size_t GetEdgeNum() const override { return oenum_ + extra_oenum_; }

  vid_t GetVerticesNum() const override { return tvnum_; }

  size_t GetTotalVerticesNum() const override {
    return vm_ptr_->GetTotalVertexSize();
  }

  bool GetVertex(const oid_t& oid, vertex_t& v) const override {
    vid_t gid;
    if (vm_ptr_->GetGid(oid, gid)) {
      return ((gid >> fid_offset_) == fid_) ? InnerVertexGid2Vertex(gid, v)
                                            : OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  oid_t GetId(const vertex_t& v) const override {
    return (IsInnerVertex(v)) ? GetInnerVertexId(v) : GetOuterVertexId(v);
  }

  const vdata_t& GetData(const vertex_t& v) const override {
    return ivdata_[v.GetValue()];
  }

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

  const std::vector<vid_t>& GetOuterVerticesGid() const { return ovgid_; }

  bool Gid2Vertex(const vid_t& gid, vertex_t& v) const override {
    return ((gid >> fid_offset_) == fid_) ? InnerVertexGid2Vertex(gid, v)
                                          : OuterVertexGid2Vertex(gid, v);
  }

  vid_t Vertex2Gid(const vertex_t& v) const override {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

  bool Oid2Gid(const oid_t& oid, vid_t& gid) const {
    return vm_ptr_->GetGid(oid, gid);
  }

  bool Oid2Gid(fid_t fid, const oid_t& oid, vid_t& gid) const {
    return vm_ptr_->GetGid(fid, oid, gid);
  }

  oid_t Gid2Oid(const vid_t& gid) const {
    OID_T oid;
    vm_ptr_->GetOid(gid, oid);
    return oid;
  }

  vid_t GetInnerVerticesNum() const override { return ivnum_; }

  vid_t GetOuterVerticesNum() const override { return ovnum_; }

  vertex_range_t InnerVertices() const override {
    return vertex_range_t(0, ivnum_);
  }

  vertex_range_t OuterVertices() const override {
    return vertex_range_t(id_mask_ - ovnum_ + 1, id_mask_ + 1);
  }

  vertex_range_t OuterVertices(fid_t fid) const { return vertex_range_t(0, 0); }

  vertex_range_t Vertices() const override { return vertex_range_t(0, 0); }

  bool IsInnerVertex(const vertex_t& v) const override {
    return (v.GetValue() < ivnum_);
  }

  bool IsInnerVertex(const vid_t& v) const { return (v < ivnum_); }

  bool IsOuterVertex(const vertex_t& v) const override {
    return (v.GetValue() >= ivnum_);
  }

  bool GetInnerVertex(const oid_t& oid, vertex_t& v) const override {
    vid_t gid;
    bool res = vm_ptr_->GetGid(oid, gid);
    if (res && ((gid >> fid_offset_) == fid_)) {
      return InnerVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  bool GetOuterVertex(const oid_t& oid, vertex_t& v) const override {
    vid_t gid;
    if (vm_ptr_->GetGid(oid, gid)) {
      return OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  oid_t GetInnerVertexId(const vertex_t& v) const override {
    oid_t oid;
    vm_ptr_->GetOid(fid_, v.GetValue(), oid);
    return oid;
  }

  oid_t GetOuterVertexId(const vertex_t& v) const override {
    vid_t gid = ovgid_[id_mask_ - v.GetValue()];
    oid_t oid;
    vm_ptr_->GetOid(gid, oid);
    return oid;
  }

  bool InnerVertexGid2Vertex(const vid_t& gid, vertex_t& v) const override {
    v.SetValue(gid & id_mask_);
    return true;
  }

  bool OuterVertexGid2Vertex(const vid_t& gid, vertex_t& v) const override {
    auto iter = ovg2i_.find(gid);
    if (iter != ovg2i_.end()) {
      v.SetValue(id_mask_ - iter->second);
      return true;
    } else {
      return false;
    }
  }

  vid_t GetOuterVertexGid(const vertex_t& v) const override {
    return ovgid_[id_mask_ - v.GetValue()];
  }

  vid_t GetInnerVertexGid(const vertex_t& v) const override {
    return (v.GetValue() | ((vid_t) fid_ << fid_offset_));
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

  void SetupMirrorInfo(fid_t fid, const VertexRange<vid_t>& range,
                       const std::vector<vid_t>& gid_list) {}

 private:
  vid_t lid2Index(const vid_t& lid) const {
    return (lid < ivnum_) ? lid : (id_mask_ - lid + ivnum_);
  }

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
  vid_t ivnum_, ovnum_, tvnum_, id_mask_;
  vid_t max_old_ilid_, min_old_olid_;
  size_t oenum_;
  size_t extra_oenum_;
  int fid_offset_;
  fid_t fid_, fnum_;

  ska::flat_hash_map<vid_t, vid_t> ovg2i_;
  std::vector<vid_t> ovgid_;
  std::vector<vdata_t> ivdata_;
  Array<nbr_t, Allocator<nbr_t>> oe_;
  Array<nbr_t*, Allocator<nbr_t*>> oeoffset_;
  Array<int, Allocator<int>> extra_oe_;
  NbrMapSpace<vid_t, edata_t> extra_edge_space_;

  std::unique_ptr<FragmentIndicesBase<oid_t, vid_t, vdata_t, edata_t>>
      fragment_indices_;
};

}  // namespace grape
#endif  // EXAMPLES_GNN_SAMPLER_APPEND_ONLY_EDGECUT_FRAGMENT_H_
