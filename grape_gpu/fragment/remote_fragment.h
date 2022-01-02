
#ifndef GRAPEGPU_GRAPE_GPU_FRAGMENT_REMOTE_FRAGMENT_H_
#define GRAPEGPU_GRAPE_GPU_FRAGMENT_REMOTE_FRAGMENT_H_
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iosfwd>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cuda_hashmap/hash_map.h"
#include "examples/flags.h"
#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/types.h"
#include "grape/util.h"
#include "grape/utils/vertex_array.h"
#include "grape/vertex_map/global_vertex_map.h"
#include "grape_gpu/config.h"
#include "grape_gpu/fragment/coo_fragment.h"
#include "grape_gpu/fragment/device_fragment.h"
#include "grape_gpu/fragment/host_fragment.h"
#include "grape_gpu/fragment/id_parser.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/ipc_array.h"
#include "grape_gpu/vertex_map/device_vertex_map.h"

namespace grape_gpu {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          grape::LoadStrategy _load_strategy = grape::LoadStrategy::kOnlyOut>
class RemoteFragment {
 public:
  using internal_vertex_t = grape::internal::Vertex<VID_T, VDATA_T>;
  using edge_t = grape::Edge<VID_T, EDATA_T>;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using vertex_t = Vertex<VID_T>;
  using const_adj_list_t = ConstAdjList<VID_T, EDATA_T>;
  using adj_list_t = AdjList<VID_T, EDATA_T>;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_range_t = VertexRange<VID_T>;
  template <typename DATA_T>
  using vertex_array_t = grape::VertexArray<DATA_T, vid_t>;
  using vertex_map_t = grape::GlobalVertexMap<oid_t, vid_t>;
  using dev_vertex_map_t = grape_gpu::DeviceVertexMap<oid_t, vid_t>;
  using device_t =
      dev::DeviceFragment<OID_T, VID_T, VDATA_T, EDATA_T, _load_strategy>;
  using coo_t = COOFragment<oid_t, vid_t, vdata_t, edata_t>;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

  static constexpr grape::LoadStrategy load_strategy = _load_strategy;

  RemoteFragment() = default;

  explicit RemoteFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : vm_ptr_(vm_ptr),
        d_vm_ptr_(std::make_shared<dev_vertex_map_t>(vm_ptr)) {}

  virtual ~RemoteFragment() {}

  void Init(fid_t fid, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) {
    fid_ = fid;
    fnum_ = vm_ptr_->GetFragmentNum();

    id_parser_.Init(fnum_);

    ivnum_ = vm_ptr_->GetInnerVertexSize(fid);
    tvnum_ = ivnum_;
    oenum_ = 0;
    ienum_ = 0;

    ie_.clear();
    oe_.clear();
    ieoffset_.clear();
    oeoffset_.clear();

    VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    auto is_iv_gid = [this](VID_T id) { return id_parser_.GetFid(id) == fid_; };
    {
      std::vector<VID_T> outer_vertices;
      auto first_iter_in = [&is_iv_gid, invalid_vid](
                               grape::Edge<VID_T, EDATA_T>& e,
                               std::vector<VID_T>& outer_vertices) {
        if (is_iv_gid(e.dst())) {
          if (!is_iv_gid(e.src())) {
            outer_vertices.push_back(e.src());
          }
        } else {
          e.SetEndpoint(invalid_vid, invalid_vid);
        }
      };
      auto first_iter_out = [&is_iv_gid, invalid_vid](
                                grape::Edge<VID_T, EDATA_T>& e,
                                std::vector<VID_T>& outer_vertices) {
        if (is_iv_gid(e.src())) {
          if (!is_iv_gid(e.dst())) {
            outer_vertices.push_back(e.dst());
          }
        } else {
          e.SetEndpoint(invalid_vid, invalid_vid);
        }
      };
      auto first_iter_out_in = [&is_iv_gid, invalid_vid](
                                   grape::Edge<VID_T, EDATA_T>& e,
                                   std::vector<VID_T>& outer_vertices) {
        if (is_iv_gid(e.src())) {
          if (!is_iv_gid(e.dst())) {
            outer_vertices.push_back(e.dst());
          }
        } else if (is_iv_gid(e.dst())) {
          outer_vertices.push_back(e.src());
        } else {
          e.SetEndpoint(invalid_vid, invalid_vid);
        }
      };

      if (load_strategy == grape::LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          first_iter_in(e, outer_vertices);
        }
      } else if (load_strategy == grape::LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          first_iter_out(e, outer_vertices);
        }
      } else if (load_strategy == grape::LoadStrategy::kBothOutIn) {
        for (auto& e : edges) {
          first_iter_out_in(e, outer_vertices);
        }
      } else {
        LOG(FATAL) << "Invalid load strategy";
      }

      grape::DistinctSort(outer_vertices);

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
        return (id_parser_.GetFid(gid) == fid_) ? id_parser_.GetLid(gid)
                                                : (ovg2l_.at(gid));
      };

      auto iv_gid_to_lid = [this](VID_T gid) { return id_parser_.GetLid(gid); };
      auto ov_gid_to_lid = [this](VID_T gid) { return ovg2l_.at(gid); };

      auto second_iter_in = [this, &iv_gid_to_lid, &ov_gid_to_lid, invalid_vid,
                             &is_iv_gid](grape::Edge<VID_T, EDATA_T>& e,
                                         std::vector<int>& idegree,
                                         std::vector<int>& odegree) {
        if (e.src() != invalid_vid) {
          VID_T src_lid, dst_lid = iv_gid_to_lid(e.dst());
          if (is_iv_gid(e.src())) {
            src_lid = iv_gid_to_lid(e.src());
          } else {
            src_lid = ov_gid_to_lid(e.src());
            ++odegree[src_lid];
            ++oenum_;
          }
          ++idegree[dst_lid];
          ++ienum_;
          e.SetEndpoint(src_lid, dst_lid);
        }
      };

      auto second_iter_out = [this, &iv_gid_to_lid, &ov_gid_to_lid, invalid_vid,
                              &is_iv_gid](grape::Edge<VID_T, EDATA_T>& e,
                                          std::vector<int>& idegree,
                                          std::vector<int>& odegree) {
        if (e.src() != invalid_vid) {
          VID_T src_lid = iv_gid_to_lid(e.src()), dst_lid;
          if (is_iv_gid(e.dst())) {
            dst_lid = iv_gid_to_lid(e.dst());
          } else {
            dst_lid = ov_gid_to_lid(e.dst());
            ++idegree[dst_lid];
            ++ienum_;
          }
          ++odegree[src_lid];
          ++oenum_;
          e.SetEndpoint(src_lid, dst_lid);
        }
      };

      auto second_iter_out_in = [this, &gid_to_lid, invalid_vid](
                                    grape::Edge<VID_T, EDATA_T>& e,
                                    std::vector<int>& idegree,
                                    std::vector<int>& odegree) {
        if (e.src() != invalid_vid) {
          VID_T src_lid = gid_to_lid(e.src()), dst_lid = gid_to_lid(e.dst());
          ++odegree[src_lid];
          ++idegree[dst_lid];
          ++oenum_;
          ++ienum_;
          e.SetEndpoint(src_lid, dst_lid);
        }
      };

      if (load_strategy == grape::LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          second_iter_in(e, idegree, odegree);
        }
      } else if (load_strategy == grape::LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          second_iter_out(e, idegree, odegree);
        }
      } else if (load_strategy == grape::LoadStrategy::kBothOutIn) {
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
      grape::Array<nbr_t*, grape::Allocator<nbr_t*>> ieiter(ieoffset_),
          oeiter(oeoffset_);

      auto third_iter_in =
          [invalid_vid, this](
              const grape::Edge<VID_T, EDATA_T>& e,
              grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& ieiter,
              grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& oeiter) {
            if (e.src() != invalid_vid) {
              ieiter[e.dst()]->GetEdgeSrc(e);
              ++ieiter[e.dst()];
              if (e.src() >= ivnum_) {
                oeiter[e.src()]->GetEdgeDst(e);
                ++oeiter[e.src()];
              }
            }
          };

      auto third_iter_out =
          [invalid_vid, this](
              const grape::Edge<VID_T, EDATA_T>& e,
              grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& ieiter,
              grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& oeiter) {
            if (e.src() != invalid_vid) {
              oeiter[e.src()]->GetEdgeDst(e);
              ++oeiter[e.src()];
              if (e.dst() >= ivnum_) {
                ieiter[e.dst()]->GetEdgeSrc(e);
                ++ieiter[e.dst()];
              }
            }
          };

      auto third_iter_out_in =
          [invalid_vid](
              const grape::Edge<VID_T, EDATA_T>& e,
              grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& ieiter,
              grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& oeiter) {
            if (e.src() != invalid_vid) {
              ieiter[e.dst()]->GetEdgeSrc(e);
              ++ieiter[e.dst()];
              oeiter[e.src()]->GetEdgeDst(e);
              ++oeiter[e.src()];
            }
          };

      if (load_strategy == grape::LoadStrategy::kOnlyIn) {
        for (auto& e : edges) {
          third_iter_in(e, ieiter, oeiter);
        }
      } else if (load_strategy == grape::LoadStrategy::kOnlyOut) {
        for (auto& e : edges) {
          third_iter_out(e, ieiter, oeiter);
        }
      } else if (load_strategy == grape::LoadStrategy::kBothOutIn) {
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
        if (is_iv_gid(gid)) {
          vdata_[id_parser_.GetLid(gid)] = v.vdata();
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

    __allocate_device_fragment__();
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];

    snprintf(fbuf, sizeof(fbuf), grape::kSerializationFilenameFormat,
             prefix.c_str(), fid_);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    grape::InArchive ia;

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
    snprintf(fbuf, sizeof(fbuf), grape::kSerializationFilenameFormat,
             prefix.c_str(), fid);
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    grape::OutArchive oa;
    int ils;

    CHECK(io_adaptor->ReadArchive(oa));

    oa >> ivnum_ >> ovnum_ >> ienum_ >> oenum_ >> fid_ >> fnum_ >> ils;
    auto got_load_strategy = grape::LoadStrategy(ils);

    if (got_load_strategy != load_strategy) {
      LOG(FATAL) << "load strategy not consistent.";
    }
    tvnum_ = ivnum_ + ovnum_;

    oa.Clear();

    ovgid_.clear();
    ovgid_.resize(ovnum_);
    if (ovnum_ > 0) {
      CHECK(io_adaptor->Read(&ovgid_[0], ovnum_ * sizeof(VID_T)));
    }

    id_parser_.Init(fnum_);
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

    __allocate_device_fragment__();
  }

  void PrepareToRunApp(grape::MessageStrategy strategy, bool need_split_edges,
                       bool need_build_device_vm) {
    Stream stream;
    if (strategy == grape::MessageStrategy::kAlongEdgeToOuterVertex ||
        strategy == grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex ||
        strategy == grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initMessageDestination(stream, strategy);
    }

    if (need_split_edges) {
      if (load_strategy == grape::LoadStrategy::kOnlyIn ||
          load_strategy == grape::LoadStrategy::kBothOutIn) {
        __init_edges_splitter__(stream, ieoffset_, iespliters_,
                                d_ieoffset_[local_id_], d_iespliters_holder_,
                                d_iespliters_);
      }
      if (load_strategy == grape::LoadStrategy::kOnlyOut ||
          load_strategy == grape::LoadStrategy::kBothOutIn) {
        __init_edges_splitter__(stream, oeoffset_, oespliters_,
                                d_oeoffset_[local_id_], d_oespliters_holder_,
                                d_oespliters_);
      }
    }

    if (need_build_device_vm) {
      d_vm_ptr_->Init(stream);
    }
    stream.Sync();
  }

  inline fid_t fid() const { return fid_; }

  inline fid_t fnum() const { return fnum_; }

  inline const vid_t* GetOuterVerticesGid() const { return &ovgid_[0]; }

  inline size_t GetEdgeNum() const { return ienum_ + oenum_; }

  inline VID_T GetVerticesNum() const { return tvnum_; }

  size_t GetTotalVerticesNum() const { return vm_ptr_->GetTotalVertexSize(); }

  inline vertex_range_t Vertices() const { return vertex_range_t(0, tvnum_); }

  inline vertex_range_t LocalVertices(int local_id) const {
    return vertex_range_t(0, tvnums_[local_id]);
  }

  inline vertex_range_t InnerVertices() const {
    return vertex_range_t(0, ivnum_);
  }

  inline vertex_range_t LocalInnerVertices(int local_id) const {
    return vertex_range_t(0, ivnums_[local_id]);
  }

  inline vertex_range_t LocalOuterVertices(int local_id) const {
    return vertex_range_t(ivnums_[local_id], ovnums_[local_id]);
  }

  inline vertex_range_t OuterVertices() const {
    return vertex_range_t(ivnum_, tvnum_);
  }

  inline vertex_range_t OuterVertices(fid_t fid) const {
    return outer_vertices_of_frag_[fid];
  }

  inline bool GetVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      return id_parser_.GetFid(gid) == fid_ ? InnerVertexGid2Vertex(gid, v)
                                            : OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  inline OID_T GetId(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexId(v) : GetOuterVertexId(v);
  }

  inline fid_t GetFragId(const vertex_t& u) const {
    auto gid = ovgid_[u.GetValue() - ivnum_];
    return IsInnerVertex(u) ? fid_ : id_parser_.GetFid(gid);
  }

  inline const VDATA_T& GetData(const vertex_t& v) const {
    return vdata_[v.GetValue()];
  }

  inline void SetData(const vertex_t& v, const VDATA_T& val) {
    vdata_[v.GetValue()] = val;
  }

  inline bool HasChild(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    return oeoffset_[v.GetValue()] != oeoffset_[v.GetValue() + 1];
  }

  inline bool HasParent(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    return ieoffset_[v.GetValue()] != ieoffset_[v.GetValue() + 1];
  }

  inline int GetLocalOutDegree(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    return oeoffset_[v.GetValue() + 1] - oeoffset_[v.GetValue()];
  }

  inline int GetLocalInDegree(const vertex_t& v) const {
    assert(IsInnerVertex(v));
    return ieoffset_[v.GetValue() + 1] - ieoffset_[v.GetValue()];
  }

  inline bool Gid2Vertex(const VID_T& gid, vertex_t& v) const {
    return id_parser_.GetFid(gid) == fid_ ? InnerVertexGid2Vertex(gid, v)
                                          : OuterVertexGid2Vertex(gid, v);
  }

  inline VID_T Vertex2Gid(const vertex_t& v) const {
    return IsInnerVertex(v) ? GetInnerVertexGid(v) : GetOuterVertexGid(v);
  }

  inline VID_T GetInnerVerticesNum() const { return ivnum_; }

  inline VID_T GetOuterVerticesNum() const { return ovnum_; }

  inline bool IsInnerVertex(const vertex_t& v) const {
    return (v.GetValue() < ivnum_);
  }

  inline bool IsOuterVertex(const vertex_t& v) const {
    return (v.GetValue() < tvnum_ && v.GetValue() >= ivnum_);
  }

  inline bool GetInnerVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      if (id_parser_.GetFid(gid) == fid_) {
        v.SetValue(id_parser_.GetLid(gid));
        return true;
      }
    }
    return false;
  }

  inline bool GetOuterVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      return OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  inline OID_T GetInnerVertexId(const vertex_t& v) const {
    OID_T internal_oid;
    CHECK(vm_ptr_->GetOid(fid_, v.GetValue(), internal_oid));
    return internal_oid;
  }

  inline OID_T GetOuterVertexId(const vertex_t& v) const {
    VID_T gid = ovgid_[v.GetValue() - ivnum_];
    OID_T internal_oid;
    CHECK(vm_ptr_->GetOid(gid, internal_oid));
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

  inline bool InnerVertexGid2Vertex(const VID_T& gid, vertex_t& v) const {
    v.SetValue(id_parser_.GetLid(gid));
    return true;
  }

  inline bool OuterVertexGid2Vertex(const VID_T& gid, vertex_t& v) const {
    auto iter = ovg2l_.find(gid);
    if (iter != ovg2l_.end()) {
      v.SetValue(iter->second);
      return true;
    } else {
      return false;
    }
  }

  inline VID_T GetOuterVertexGid(const vertex_t& v) const {
    return ovgid_[v.GetValue() - ivnum_];
  }
  inline VID_T GetInnerVertexGid(const vertex_t& v) const {
    return id_parser_.Lid2Gid(fid_, v.GetValue());
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
  inline DestList IEDests(const vertex_t& v) const {
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
  inline DestList OEDests(const vertex_t& v) const {
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
  inline DestList IOEDests(const vertex_t& v) const {
    assert(!iodoffset_.empty());
    assert(IsInnerVertex(v));
    return DestList(iodoffset_[v.GetValue()], iodoffset_[v.GetValue() + 1]);
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
  inline adj_list_t GetIncomingAdjList(const vertex_t& v) {
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
  inline const_adj_list_t GetIncomingAdjList(const vertex_t& v) const {
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
  inline adj_list_t GetOutgoingAdjList(const vertex_t& v) {
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
  inline const_adj_list_t GetOutgoingAdjList(const vertex_t& v) const {
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
  inline adj_list_t GetIncomingInnerVertexAdjList(const vertex_t& v) {
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
      const vertex_t& v) const {
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
  inline adj_list_t GetIncomingOuterVertexAdjList(const vertex_t& v) {
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
      const vertex_t& v) const {
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
  inline adj_list_t GetOutgoingInnerVertexAdjList(const vertex_t& v) {
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
      const vertex_t& v) const {
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
  inline adj_list_t GetOutgoingOuterVertexAdjList(const vertex_t& v) {
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
      const vertex_t& v) const {
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

  std::shared_ptr<vertex_map_t> vm_ptr() const { return vm_ptr_; }

  inline vid_t ConsecutiveGid2Gid(vid_t consecutive_gid) const {
    fid_t fid = std::lower_bound(ps_ivnum_.begin(), ps_ivnum_.end(),
                                 consecutive_gid + 1) -
                ps_ivnum_.begin() - 1;
    vid_t begin_offset = ps_ivnum_[fid];
    vid_t lid = consecutive_gid - begin_offset;

    return id_parser_.Lid2Gid(fid, lid);
  }

  void SetupMirrorInfo(fid_t fid, const vertex_range_t& range,
                       const std::vector<VID_T>& gid_list) {
    mirrors_range_[fid].SetRange(range.begin().GetValue(),
                                 range.end().GetValue());
    auto& vertex_vec = mirrors_of_frag_[fid];
    vertex_vec.resize(gid_list.size());

    for (size_t i = 0; i < gid_list.size(); ++i) {
      auto gid = gid_list[i];
      CHECK_EQ(id_parser_.GetFid(gid), fid_);
      vertex_vec[i].SetValue(id_parser_.GetLid(gid));
    }
    auto& comm_spec = vm_ptr_->GetCommSpec();
    // SetupMirrorInfo will be invoked after Init, so we copy mirrors for here
    d_mirrors_of_frag_holder_[fid] = mirrors_of_frag_[fid];
    d_mirrors_of_frag_[fid] =
        ArrayView<vertex_t>(d_mirrors_of_frag_holder_[fid]);
  }

  device_t DeviceObject() const {
    device_t dev_frag;

    dev_frag.vm_ = d_vm_ptr_->DeviceObject();

    dev_frag.ivnum_ = ivnum_;
    dev_frag.ovnum_ = ovnum_;
    dev_frag.tvnum_ = tvnum_;
    dev_frag.ienum_ = ienum_;
    dev_frag.oenum_ = oenum_;

    dev_frag.fid_ = fid_;

    dev_frag.id_parser_ = id_parser_;

    dev_frag.ovg2l_ = d_ovg2l_.get();
    dev_frag.ovgid_ = ArrayView<vid_t>(d_ovgid_[local_id_]);

    dev_frag.ieoffset_ = ArrayView<nbr_t*>(d_ieoffset_[local_id_]);
    dev_frag.oeoffset_ = ArrayView<nbr_t*>(d_oeoffset_[local_id_]);

    if (d_ie_ != nullptr) {
      dev_frag.ie_ = d_ie_->view(local_id_);
    }

    if (d_oe_ != nullptr) {
      dev_frag.oe_ = d_oe_->view(local_id_);
    }

    dev_frag.vdata_ = ArrayView<VDATA_T>(d_vdata_);

    dev_frag.idst_ = ArrayView<fid_t>(d_idst_);
    dev_frag.odst_ = ArrayView<fid_t>(d_odst_);
    dev_frag.iodst_ = ArrayView<fid_t>(d_iodst_);

    dev_frag.idoffset_ = ArrayView<fid_t*>(d_idoffset_);
    dev_frag.odoffset_ = ArrayView<fid_t*>(d_odoffst_);
    dev_frag.iodoffset_ = ArrayView<fid_t*>(d_iodoffset_);

    dev_frag.iespliters_ = ArrayView<ArrayView<nbr_t*>>(d_iespliters_);
    dev_frag.oespliters_ = ArrayView<ArrayView<nbr_t*>>(d_oespliters_);

    dev_frag.outer_vertices_of_frag_ =
        ArrayView<vertex_range_t>(d_outer_vertices_of_frag_);
    dev_frag.mirrors_of_frag_ =
        ArrayView<ArrayView<vertex_t>>(d_mirrors_of_frag_);

    dev_frag.ps_ivnum_ = ArrayView<vid_t>(d_ps_ivnum_);

    return dev_frag;
  }

  device_t DeviceObject(int local_id) const {
    device_t dev_frag;

    dev_frag.vm_ = d_vm_ptr_->DeviceObject();

    dev_frag.ivnum_ = ivnums_[local_id];
    dev_frag.ovnum_ = ovnums_[local_id];
    dev_frag.tvnum_ = tvnums_[local_id];
    dev_frag.ienum_ = ienums_[local_id];
    dev_frag.oenum_ = oenums_[local_id];

    dev_frag.fid_ = fids_[local_id];

    dev_frag.id_parser_ = id_parser_;

    dev_frag.ovg2l_ = d_ovg2l_.get();
    dev_frag.ovgid_ = ArrayView<vid_t>(d_ovgid_[local_id]);

    dev_frag.in_degree_ = ArrayView<size_t>(d_in_degree_[local_id]);
    dev_frag.out_degree_ = ArrayView<size_t>(d_out_degree_[local_id]);
    dev_frag.ieoffset_ = ArrayView<nbr_t*>(d_ieoffset_[local_id]);
    dev_frag.oeoffset_ = ArrayView<nbr_t*>(d_oeoffset_[local_id]);

    dev_frag.ps_ivnum_ = ArrayView<vid_t>(d_ps_ivnum_);

    return dev_frag;
  }

  ArrayView<device_t> DeviceObjects() const {
    return ArrayView<device_t>(dev_objs_);
  }

  virtual void __allocate_device_fragment__() {
    double mirror_factor = FLAGS_ws_mirror_factor;
    Stream stream;
    auto comm_spec = vm_ptr_->GetCommSpec();
    grape::Communicator communicator;
    auto local_num = comm_spec.local_num();
    auto compute_prefix_sum =
        [](const grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& offset) {
          pinned_vector<size_t> prefix_sum(offset.size());

          for (size_t idx = 0; idx < prefix_sum.size(); idx++) {
            prefix_sum[idx] = offset[idx] - offset[0];
          }
          return prefix_sum;
        };

    local_id_ = comm_spec.local_id();
    d_ieoffset_.resize(local_num);
    d_oeoffset_.resize(local_num);

    // gather metadata on the same machine
    communicator.InitCommunicator(comm_spec.local_comm());
    communicator.template AllGather(ivnum_, ivnums_);
    communicator.template AllGather(ovnum_, ovnums_);
    communicator.template AllGather(tvnum_, tvnums_);
    communicator.template AllGather(ienum_, ienums_);
    communicator.template AllGather(oenum_, oenums_);
    communicator.template AllGather(fid_, fids_);

    auto calc_degree = [&](const pinned_vector<size_t>& prefix_sum) {
      std::vector<size_t> degree(prefix_sum.size() - 1);

      for (size_t idx = 0; idx < degree.size(); idx++) {
        degree[idx] = prefix_sum[idx + 1] - prefix_sum[idx];
      }
      return degree;
    };

    // calculate an array shows that whether the edges of inner vertex are
    // cached currently, we only implemented static cache according to degree
    auto calc_cached = [&](const pinned_vector<size_t>& prefix_sum) {
      std::vector<std::pair<size_t, size_t>> idx_degree(prefix_sum.size() - 1);

      for (size_t idx = 0; idx < idx_degree.size(); idx++) {
        auto degree = prefix_sum[idx + 1] - prefix_sum[idx];

        idx_degree[idx] = std::make_pair(idx, degree);
      }

      // sort vertices by degree in desc order
      std::sort(idx_degree.begin(), idx_degree.end(),
                [](const std::pair<size_t, size_t>& lhs,
                   const std::pair<size_t, size_t>& rhs) {
                  return lhs.second > rhs.second;
                });
      pinned_vector<int> cached(idx_degree.size());
      // cache top "mirror_factor" percentage vertices with greater degree
      for (size_t idx = 0; idx < idx_degree.size(); idx++) {
        cached[idx_degree[idx].first] =
            (idx < idx_degree.size() * mirror_factor) ? 1 : 0;
      }
      return cached;
    };

    // we calculate two degree arrays used for indexing end pointer in CSR/CSC
    d_in_degree_.resize(local_num);
    d_out_degree_.resize(local_num);

    if (load_strategy == grape::LoadStrategy::kOnlyIn ||
        load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_ie_ = std::make_shared<IPCArray<nbr_t, IPCMemoryPlacement::kDevice>>(
          comm_spec);
      d_ie_->Init(ienum_);
      d_ieoffset_views_.resize(local_num);

      auto d_ie_view = d_ie_->view(comm_spec.local_id());

      CHECK_CUDA(cudaMemcpyAsync(d_ie_view.data(), ie_.data(),
                                 sizeof(nbr_t) * ienum_, cudaMemcpyHostToDevice,
                                 stream.cuda_stream()));
      stream.Sync();
      MPI_Barrier(comm_spec.local_comm());
      // if we enable cache, sync edges to mirrors
      if (mirror_factor > 0) {
        d_ie_->SyncToMirror();
      }

      // this offset is represent by index
      auto prefix_sum = compute_prefix_sum(ieoffset_);
      std::vector<pinned_vector<size_t>> prefix_sum_vec;

      // now, we have every workers' row offset array
      communicator.template AllGather(prefix_sum, prefix_sum_vec);

      // exchange offset;
      for (int local_id = 0; local_id < local_num; local_id++) {
        auto& curr_prefix_sum = prefix_sum_vec[local_id];
        auto offset_size = curr_prefix_sum.size();  // tvnum + 1

        d_ieoffset_[local_id].resize(offset_size);
        // Calculating standard version of row offset if we are on local or
        // cache function is disabled
        if (local_id == comm_spec.local_id() || mirror_factor == 0) {
          auto ie_view = d_ie_->view(local_id);

          CalculateOffsetWithPrefixSum<nbr_t, size_t>(
              stream, ArrayView<size_t>(curr_prefix_sum), ie_view.data(),
              thrust::raw_pointer_cast(d_ieoffset_[local_id].data()));
        } else {
          // Calculate row offset for mirrors, Refer device_fragment.h:400
          // We can implement *dynamic cache* by just changing the pointer of
          // d_ieoffset_[fid][lid] from master/mirror to mirror/master pointer
          // Refer: dev_utils.h:63
          auto master_ie_view = d_ie_->view(local_id);
          auto mirror_ie_view = d_ie_->mirror_view(local_id);
          auto cached = calc_cached(curr_prefix_sum);

          CalculateOffsetWithPrefixSum<nbr_t, size_t>(
              stream, ArrayView<size_t>(curr_prefix_sum), cached.data(),
              master_ie_view.data(), mirror_ie_view.data(),
              thrust::raw_pointer_cast(d_ieoffset_[local_id].data()));
          d_in_degree_[local_id] = calc_degree(curr_prefix_sum);
        }
        d_ieoffset_views_[local_id] = ArrayView<nbr_t*>(d_ieoffset_[local_id]);
      }
    }

    if (load_strategy == grape::LoadStrategy::kOnlyOut ||
        load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_oe_ = std::make_shared<IPCArray<nbr_t, IPCMemoryPlacement::kDevice>>(
          comm_spec);
      d_oe_->Init(oenum_);
      d_oeoffset_views_.resize(local_num);

      auto d_oe_view = d_oe_->view(comm_spec.local_id());

      CHECK_CUDA(cudaMemcpyAsync(d_oe_view.data(), oe_.data(),
                                 sizeof(nbr_t) * oenum_, cudaMemcpyHostToDevice,
                                 stream.cuda_stream()));
      stream.Sync();
      MPI_Barrier(comm_spec.local_comm());
      if (mirror_factor > 0) {
        d_oe_->SyncToMirror();
      }

      // this offset is represent by index
      auto prefix_sum = compute_prefix_sum(oeoffset_);
      std::vector<pinned_vector<size_t>> prefix_sum_vec;
      communicator.template AllGather(prefix_sum, prefix_sum_vec);

      // exchange offset;
      for (int local_id = 0; local_id < local_num; local_id++) {
        auto& curr_prefix_sum = prefix_sum_vec[local_id];
        auto offset_size = curr_prefix_sum.size();  // tvnum + 1

        d_oeoffset_[local_id].resize(offset_size);

        // processing local or mirror is disabled
        if (local_id == comm_spec.local_id() || mirror_factor == 0) {
          auto oe_view = d_oe_->view(local_id);

          CalculateOffsetWithPrefixSum<nbr_t, size_t>(
              stream, ArrayView<size_t>(curr_prefix_sum), oe_view.data(),
              thrust::raw_pointer_cast(d_oeoffset_[local_id].data()));
        } else {
          auto master_oe_view = d_oe_->view(local_id);
          auto mirror_oe_view = d_oe_->mirror_view(local_id);
          auto cached = calc_cached(curr_prefix_sum);

          CalculateOffsetWithPrefixSum<nbr_t, size_t>(
              stream, ArrayView<size_t>(curr_prefix_sum), cached.data(),
              master_oe_view.data(), mirror_oe_view.data(),
              thrust::raw_pointer_cast(d_oeoffset_[local_id].data()));
          d_out_degree_[local_id] = calc_degree(curr_prefix_sum);
        }
        d_oeoffset_views_[local_id] = ArrayView<nbr_t*>(d_oeoffset_[local_id]);
      }
    }

    if (sizeof(internal_vertex_t) > sizeof(VID_T)) {
      d_vdata_.resize(tvnum_);
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_vdata_.data()),
                                 vdata_.data(), sizeof(VDATA_T) * tvnum_,
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));
    }

    std::vector<grape::Array<VID_T, grape::Allocator<VID_T>>> local_ovgids;

    // Gather all ov lid->ov gid arrays for message passing
    communicator.template AllGather(ovgid_, local_ovgids);
    d_ovgid_.resize(local_num);

    for (int local_id = 0; local_id < local_num; local_id++) {
      auto& ovgid = local_ovgids[local_id];
      auto& d_ovgid = d_ovgid_[local_id];
      auto ovnum = ovgid.size();

      d_ovgid.resize(ovnum);
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_ovgid.data()),
                                 ovgid.data(), sizeof(VID_T) * ovnum,
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));
    }

    {
      auto size = ovg2l_.size();
      pinned_vector<VID_T> gids(size);
      pinned_vector<VID_T> lids(size);
      size_t idx = 0;

      d_ovg2l_ = std::shared_ptr<CUDASTL::HashMap<VID_T, VID_T>>(
          CUDASTL::CreateHashMap<VID_T, VID_T, CUDASTL::HashFunc<VID_T>>(
              stream.cuda_stream(), ovg2l_.bucket_count(), size),
          [this](CUDASTL::HashMap<VID_T, VID_T>* hash_map) {
            CUDASTL::DestroyHashMap(hash_map);
          });

      for (auto& gl : ovg2l_) {
        gids[idx] = gl.first;
        lids[idx] = gl.second;
        idx++;
      }

      LaunchKernel(
          stream,
          [] __device__(VID_T * gids, VID_T * lids, VID_T size,
                        CUDASTL::HashMap<VID_T, VID_T> * ovg2l) {
            auto tid = TID_1D;
            auto nthreads = TOTAL_THREADS_1D;

            for (VID_T idx = 0 + tid; idx < size; idx += nthreads) {
              VID_T gid = gids[idx];
              VID_T lid = lids[idx];

              (*ovg2l)[gid] = lid;
            }
          },
          gids.data(), lids.data(), size, d_ovg2l_.get());
    }

    d_mirrors_of_frag_holder_.resize(fnum_);
    d_mirrors_of_frag_.resize(fnum_);
    d_outer_vertices_of_frag_.resize(fnum_);

    for (fid_t fid = 0; fid < fnum_; fid++) {
      d_mirrors_of_frag_holder_[fid] = mirrors_of_frag_[fid];
      ArrayView<vertex_t> mirror_holder(d_mirrors_of_frag_holder_[fid]);
      CHECK_CUDA(cudaMemcpyAsync(
          thrust::raw_pointer_cast(d_mirrors_of_frag_.data()) + fid,
          &mirror_holder, sizeof(ArrayView<vertex_t>), cudaMemcpyHostToDevice,
          stream.cuda_stream()));
      d_outer_vertices_of_frag_[fid] = outer_vertices_of_frag_[fid];
    }

    stream.Sync();

    VID_T offset = 0;
    for (fid_t fid = 0; fid < fnum_; fid++) {
      auto nv = vm_ptr_->GetInnerVertexSize(fid);
      ps_ivnum_.push_back(offset);
      offset += nv;
    }
    ps_ivnum_.push_back(offset);
    d_ps_ivnum_ = ps_ivnum_;

    // create device objects including master and mirrors
    dev_objs_.resize(fnum_);
    for (fid_t fid = 0; fid < fnum_; fid++) {
      dev_objs_[fid] = DeviceObject(fid);
    }

    VLOG(1) << "fid: " << fid_ << " ivnum: " << ivnum_ << " ovnum: " << ovnum_
            << " ienum: " << ienum_ << " oenum: " << oenum_;
  }

  void __init_edges_splitter__(
      const Stream& stream,
      grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& eoffset,
      std::vector<grape::Array<nbr_t*, grape::Allocator<nbr_t*>>>& espliters,
      thrust::device_vector<nbr_t*>& d_eoffset,
      std::vector<thrust::device_vector<nbr_t*>>& d_espliters_holder,
      thrust::device_vector<ArrayView<nbr_t*>>& d_espliters) {
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
          fid_t fid = id_parser_.GetFid(ovgid_[e.neighbor.GetValue() - ivnum_]);
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

    d_espliters_holder.resize(fnum_ + 1);
    for (auto& vec : d_espliters_holder) {
      vec.resize(ivnum_);
      d_espliters.push_back(ArrayView<nbr_t*>(vec));
    }
    for (fid_t fid = 0; fid < fnum_ + 1; fid++) {
      auto& e_splitter = espliters[fid];

      if (!e_splitter.empty()) {
        pinned_vector<size_t> h_degree(e_splitter.size());
        for (size_t i = 0; i < e_splitter.size(); i++) {
          h_degree[i] = e_splitter[i] - eoffset[0];
        }

        LaunchKernel(
            stream,
            [] __device__(size_t * h_degree, vid_t ivnum,
                          ArrayView<nbr_t*> offset,
                          ArrayView<nbr_t*> espliter) {
              auto tid = TID_1D;
              auto nthreads = TOTAL_THREADS_1D;

              for (size_t i = 0 + tid; i < ivnum; i += nthreads) {
                espliter[i] = offset[0] + h_degree[i];
              }
            },
            thrust::raw_pointer_cast(h_degree.data()), ivnum_,
            ArrayView<nbr_t*>(d_eoffset), ArrayView<nbr_t*>(d_espliters[fid]));
      }
    }
  }

 protected:
  void initMessageDestination(const Stream& stream,
                              const grape::MessageStrategy& msg_strategy) {
    if (msg_strategy ==
        grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      initDestFidList(stream, false, true, odst_, odoffset_, d_odst_,
                      d_odoffst_);
    } else if (msg_strategy ==
               grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      initDestFidList(stream, true, false, idst_, idoffset_, d_idst_,
                      d_idoffset_);
    } else if (msg_strategy ==
               grape::MessageStrategy::kAlongEdgeToOuterVertex) {
      initDestFidList(stream, true, true, iodst_, iodoffset_, d_iodst_,
                      d_iodoffset_);
    }
  }

  void initDestFidList(
      const Stream& stream, bool in_edge, bool out_edge,
      grape::Array<fid_t, grape::Allocator<fid_t>>& fid_list,
      grape::Array<fid_t*, grape::Allocator<fid_t*>>& fid_list_offset,
      thrust::device_vector<fid_t>& d_fid_list,
      thrust::device_vector<fid_t*>& d_fid_list_offset) {
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
            fid_t f = id_parser_.GetFid(ovgid_[lid - ivnum_]);
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
            fid_t f = id_parser_.GetFid(ovgid_[lid - ivnum_]);
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
    pinned_vector<size_t> prefix_sum(ivnum_ + 1, 0);
    ArrayView<size_t> d_prefix_sum(prefix_sum.data(), prefix_sum.size());

    for (VID_T i = 0; i < ivnum_; ++i) {
      fid_list_offset[i + 1] = fid_list_offset[i] + id_num[i];
      prefix_sum[i + 1] = prefix_sum[i] + id_num[i];
    }

    d_fid_list.resize(fid_list.size());
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_fid_list.data()),
                               fid_list.data(), sizeof(fid_t) * fid_list.size(),
                               cudaMemcpyHostToDevice, stream.cuda_stream()));

    d_fid_list_offset.resize(fid_list_offset.size());
    CalculateOffsetWithPrefixSum<fid_t, size_t>(
        stream, d_prefix_sum, thrust::raw_pointer_cast(d_fid_list.data()),
        thrust::raw_pointer_cast(d_fid_list_offset.data()));
    stream.Sync();
  }

  void initOuterVerticesOfFragment() {
    std::vector<int> frag_v_num(fnum_, 0);
    fid_t last_fid = 0;

    for (VID_T i = 0; i < ovnum_; ++i) {
      VID_T gid = ovgid_[i];
      fid_t fid = id_parser_.GetFid(gid);

      CHECK_GE(fid, last_fid);
      last_fid = fid;
      ++frag_v_num[fid];
    }

    outer_vertices_of_frag_.clear();
    outer_vertices_of_frag_.reserve(fnum_);

    VID_T cur_lid = ivnum_;

    for (fid_t fid = 0; fid < fnum_; fid++) {
      VID_T next_lid = cur_lid + frag_v_num[fid];
      outer_vertices_of_frag_.emplace_back(cur_lid, next_lid);
      cur_lid = next_lid;
    }
    CHECK_EQ(cur_lid, tvnum_);
  }

  std::shared_ptr<vertex_map_t> vm_ptr_;
  VID_T ivnum_, ovnum_, tvnum_;
  size_t ienum_{}, oenum_{};
  fid_t fid_{}, fnum_{};

  IdParser<VID_T> id_parser_;

  std::unordered_map<VID_T, VID_T> ovg2l_;
  grape::Array<VID_T, grape::Allocator<VID_T>> ovgid_;
  grape::Array<nbr_t, grape::Allocator<nbr_t>> ie_, oe_;

  grape::Array<nbr_t*, grape::Allocator<nbr_t*>> ieoffset_, oeoffset_;
  grape::Array<VDATA_T, grape::Allocator<VDATA_T>> vdata_;

  std::vector<vertex_range_t> outer_vertices_of_frag_;

  std::vector<vertex_range_t> mirrors_range_;
  std::vector<std::vector<vertex_t>> mirrors_of_frag_;

  grape::Array<fid_t, grape::Allocator<fid_t>> idst_, odst_, iodst_;
  grape::Array<fid_t*, grape::Allocator<fid_t*>> idoffset_, odoffset_,
      iodoffset_;

  std::vector<grape::Array<nbr_t*, grape::Allocator<nbr_t*>>> iespliters_,
      oespliters_;

  int local_id_;

  std::vector<VID_T> ivnums_, ovnums_, tvnums_;
  std::vector<size_t> ienums_, oenums_;
  std::vector<fid_t> fids_;

  std::shared_ptr<dev_vertex_map_t> d_vm_ptr_;
  std::shared_ptr<CUDASTL::HashMap<VID_T, VID_T>> d_ovg2l_;

  std::vector<thrust::device_vector<VID_T>> d_ovgid_;

  std::shared_ptr<IPCArray<nbr_t, IPCMemoryPlacement::kDevice>> d_ie_, d_oe_;

  std::vector<thrust::device_vector<nbr_t*>> d_ieoffset_, d_oeoffset_;
  thrust::device_vector<ArrayView<nbr_t*>> d_ieoffset_views_, d_oeoffset_views_;

  thrust::device_vector<VDATA_T> d_vdata_;

  thrust::device_vector<fid_t> d_idst_, d_odst_, d_iodst_;
  thrust::device_vector<fid_t*> d_idoffset_, d_odoffst_, d_iodoffset_;

  std::vector<thrust::device_vector<nbr_t*>> d_iespliters_holder_,
      d_oespliters_holder_;
  thrust::device_vector<ArrayView<nbr_t*>> d_iespliters_, d_oespliters_;

  thrust::device_vector<vertex_range_t> d_outer_vertices_of_frag_;

  std::vector<thrust::device_vector<vertex_t>> d_mirrors_of_frag_holder_;
  thrust::device_vector<ArrayView<vertex_t>> d_mirrors_of_frag_;

  std::vector<thrust::device_vector<size_t>> d_in_degree_, d_out_degree_;

  std::shared_ptr<coo_t> coo_frag_;

  thrust::host_vector<VID_T> ps_ivnum_;
  thrust::device_vector<VID_T> d_ps_ivnum_;

  thrust::device_vector<device_t> dev_objs_;
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_FRAGMENT_REMOTE_FRAGMENT_H_
