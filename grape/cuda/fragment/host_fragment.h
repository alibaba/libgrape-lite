/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef GRAPE_CUDA_FRAGMENT_HOST_FRAGMENT_H_
#define GRAPE_CUDA_FRAGMENT_HOST_FRAGMENT_H_

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
#include "grape/config.h"
#include "grape/cuda/fragment/coo_fragment.h"
#include "grape/cuda/fragment/device_fragment.h"
#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/dev_utils.h"
#include "grape/cuda/utils/stream.h"
#include "grape/cuda/vertex_map/device_vertex_map.h"
#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/fragment/id_parser.h"
#include "grape/fragment/immutable_edgecut_fragment.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/types.h"
#include "grape/util.h"
#include "grape/utils/vertex_array.h"
#include "grape/vertex_map/global_vertex_map.h"

namespace grape {
namespace cuda {

template <typename T, typename SIZE_T>
inline void CalculateOffsetWithPrefixSum(const Stream& stream,
                                         const ArrayView<SIZE_T>& prefix_sum,
                                         T* begin_pointer, T** offset) {
  auto size = prefix_sum.size();

  LaunchKernel(stream, [=] __device__() {
    auto tid = TID_1D;
    auto nthreads = TOTAL_THREADS_1D;

    for (size_t idx = 0 + tid; idx < size; idx += nthreads) {
      offset[idx] = begin_pointer + prefix_sum[idx];
    }
  });
}

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T,
          grape::LoadStrategy _load_strategy = grape::LoadStrategy::kOnlyOut,
          typename VERTEX_MAP_T = GlobalVertexMap<OID_T, VID_T>>
class HostFragment
    : public ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                      _load_strategy, VERTEX_MAP_T> {
 public:
  using base_t = ImmutableEdgecutFragment<OID_T, VID_T, VDATA_T, EDATA_T,
                                          _load_strategy, VERTEX_MAP_T>;
  using internal_vertex_t = typename base_t::internal_vertex_t;
  using edge_t = typename base_t::edge_t;
  using nbr_t = typename base_t::nbr_t;
  using vertex_t = typename base_t::vertex_t;
  using const_adj_list_t = typename base_t::const_adj_list_t;
  using adj_list_t = typename base_t::adj_list_t;
  using traits_t = typename base_t::traits_t;
  using vid_t = VID_T;
  using oid_t = OID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_range_t = typename base_t::vertex_range_t;

  using vertex_map_t = typename base_t::vertex_map_t;
  using dev_vertex_map_t = cuda::DeviceVertexMap<vertex_map_t>;
  using inner_vertices_t = typename base_t::inner_vertices_t;
  using outer_vertices_t = typename base_t::outer_vertices_t;
  using device_t =
      dev::DeviceFragment<OID_T, VID_T, VDATA_T, EDATA_T, _load_strategy>;
  using coo_t = COOFragment<oid_t, vid_t, vdata_t, edata_t>;

  using IsEdgeCut = std::true_type;
  using IsVertexCut = std::false_type;

  static constexpr grape::LoadStrategy load_strategy = _load_strategy;

  HostFragment() = default;

  explicit HostFragment(std::shared_ptr<vertex_map_t> vm_ptr)
      : FragmentBase<OID_T, VID_T, VDATA_T, EDATA_T, traits_t>(vm_ptr) {}

  void Init(fid_t fid, bool directed, std::vector<internal_vertex_t>& vertices,
            std::vector<edge_t>& edges) {
    base_t::Init(fid, directed, vertices, edges);
    __allocate_device_fragment__();
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    base_t::template Serialize<IOADAPTOR_T>(prefix);
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& prefix, const fid_t fid) {
    base_t::template Deserialize<IOADAPTOR_T>(prefix, fid);
    __allocate_device_fragment__();
  }

  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) {
    base_t::PrepareToRunApp(comm_spec, conf);

    Stream stream;
    if (conf.message_strategy ==
            grape::MessageStrategy::kAlongEdgeToOuterVertex ||
        conf.message_strategy ==
            grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex ||
        conf.message_strategy ==
            grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      __initMessageDestination(stream, conf.message_strategy);
    }

    if (conf.need_split_edges || conf.need_split_edges_by_fragment) {
      auto& comm_spec = vm_ptr_->GetCommSpec();
      auto& ie = ie_.get_edges();
      auto& ieoffset = ie_.get_offsets();
      auto& oe = oe_.get_edges();
      auto& oeoffset = oe_.get_offsets();
      auto offset_size = ivnum_ + ovnum_ + 1;
      auto compute_prefix_sum =
          [offset_size](
              const grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& offset) {
            pinned_vector<VID_T> prefix_sum(offset_size);

            for (vid_t idx = 0; idx < offset_size; idx++) {
              prefix_sum[idx] = offset[idx] - offset[0];
            }
            return prefix_sum;
          };
      // Engage incoming edges for outer vertices.
      if (load_strategy == grape::LoadStrategy::kOnlyOut) {
        d_ieoffset_.resize(offset_size);
        d_ie_.resize(ie_.edge_num());
        CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_ie_.data()),
                                   ie.data(), sizeof(nbr_t) * ie_.edge_num(),
                                   cudaMemcpyHostToDevice,
                                   stream.cuda_stream()));

        auto prefix_sum = compute_prefix_sum(ieoffset);
        ArrayView<VID_T> d_prefix_sum(prefix_sum.data(), prefix_sum.size());

        CalculateOffsetWithPrefixSum<nbr_t, vid_t>(
            stream, d_prefix_sum, thrust::raw_pointer_cast(d_ie_.data()),
            thrust::raw_pointer_cast(d_ieoffset_.data()));
      }
      if (load_strategy == grape::LoadStrategy::kOnlyIn) {
        d_oeoffset_.resize(offset_size);
        d_oe_.resize(oe_.edge_num());
        CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_oe_.data()),
                                   oe.data(), sizeof(nbr_t) * oe_.edge_num(),
                                   cudaMemcpyHostToDevice,
                                   stream.cuda_stream()));

        auto prefix_sum = compute_prefix_sum(oeoffset);
        ArrayView<VID_T> d_prefix_sum(prefix_sum.data(), prefix_sum.size());

        CalculateOffsetWithPrefixSum<nbr_t, vid_t>(
            stream, d_prefix_sum, thrust::raw_pointer_cast(d_oe_.data()),
            thrust::raw_pointer_cast(d_oeoffset_.data()));
      }
    }

    if (conf.need_split_edges) {
      __init_edges_splitter__(stream, ie_.get_offsets(), iespliters_,
                              d_ieoffset_, d_iespliters_holder_, d_iespliters_);
      __init_edges_splitter__(stream, oe_.get_offsets(), oespliters_,
                              d_oeoffset_, d_oespliters_holder_, d_oespliters_);
    }

    if (conf.need_split_edges_by_fragment) {
      if (load_strategy == grape::LoadStrategy::kOnlyIn ||
          load_strategy == grape::LoadStrategy::kBothOutIn) {
        __init_edges_splitter_by_fragment__(
            stream, ie_.get_offsets(), iespliters_, d_ieoffset_,
            d_iespliters_holder_, d_iespliters_);
      }
      if (load_strategy == grape::LoadStrategy::kOnlyOut ||
          load_strategy == grape::LoadStrategy::kBothOutIn) {
        __init_edges_splitter_by_fragment__(
            stream, oe_.get_offsets(), oespliters_, d_oeoffset_,
            d_oespliters_holder_, d_oespliters_);
      }
    }

    if (conf.need_mirror_info) {
      __initMirrorInfo(comm_spec);
    }

    if (conf.need_build_device_vm) {
      d_vm_ptr_->Init(stream);
    }
    stream.Sync();
  }

  using base_t::fid;
  using base_t::GetData;
  using base_t::GetEdgeNum;
  using base_t::GetFragId;
  using base_t::GetId;
  using base_t::GetIncomingAdjList;
  using base_t::GetIncomingInnerVertexAdjList;
  using base_t::GetIncomingOuterVertexAdjList;
  using base_t::GetInnerVertex;
  using base_t::GetInnerVertexGid;
  using base_t::GetInnerVertexId;
  using base_t::GetInnerVerticesNum;
  using base_t::GetLocalInDegree;
  using base_t::GetLocalOutDegree;
  using base_t::GetOuterVertexId;
  using base_t::GetOuterVerticesNum;
  using base_t::GetOutgoingAdjList;
  using base_t::GetOutgoingInnerVertexAdjList;
  using base_t::GetOutgoingOuterVertexAdjList;
  using base_t::GetTotalVerticesNum;
  using base_t::GetVertex;
  using base_t::GetVerticesNum;
  using base_t::Gid2Oid;
  using base_t::Gid2Vertex;
  using base_t::HasChild;
  using base_t::HasParent;
  using base_t::IEDests;
  using base_t::InnerVertexGid2Vertex;
  using base_t::InnerVertices;
  using base_t::IOEDests;
  using base_t::IsBorderVertex;
  using base_t::IsIncomingBorderVertex;
  using base_t::IsInnerVertex;
  using base_t::IsOuterVertex;
  using base_t::IsOutgoingBorderVertex;
  using base_t::MirrorVertices;
  using base_t::OEDests;
  using base_t::OuterVertexGid2Vertex;
  using base_t::OuterVertices;
  using base_t::SetData;
  using base_t::Vertex2Gid;
  using base_t::Vertices;

  inline const vid_t* GetOuterVerticesGid() const { return &ovgid_[0]; }

  inline bool GetOuterVertex(const OID_T& oid, vertex_t& v) const {
    VID_T gid;
    OID_T internal_oid(oid);
    if (vm_ptr_->GetGid(internal_oid, gid)) {
      return OuterVertexGid2Vertex(gid, v);
    } else {
      return false;
    }
  }

  inline bool Oid2Gid(const OID_T& oid, VID_T& gid) const {
    OID_T internal_oid(oid);
    return vm_ptr_->GetGid(internal_oid, gid);
  }

  device_t DeviceObject() const {
    device_t dev_frag;

    dev_frag.vm_ = d_vm_ptr_->DeviceObject();

    dev_frag.ivnum_ = ivnum_;
    dev_frag.ovnum_ = ovnum_;
    dev_frag.tvnum_ = ivnum_ + ovnum_;
    dev_frag.ienum_ = ie_.edge_num();
    dev_frag.oenum_ = oe_.edge_num();

    dev_frag.fid_ = fid_;

    dev_frag.id_parser_ = id_parser_;

    dev_frag.ovg2l_ = d_ovg2l_.get();
    dev_frag.ovgid_ = ArrayView<vid_t>(d_ovgid_);

    dev_frag.ieoffset_ = ArrayView<nbr_t*>(d_ieoffset_);
    dev_frag.oeoffset_ = ArrayView<nbr_t*>(d_oeoffset_);

    dev_frag.ie_ = ArrayView<nbr_t>(d_ie_);
    dev_frag.oe_ = ArrayView<nbr_t>(d_oe_);

    dev_frag.vdata_ = ArrayView<VDATA_T>(d_vdata_);

    dev_frag.idst_ = ArrayView<fid_t>(d_idst_);
    dev_frag.odst_ = ArrayView<fid_t>(d_odst_);
    dev_frag.iodst_ = ArrayView<fid_t>(d_iodst_);

    dev_frag.idoffset_ = ArrayView<fid_t*>(d_idoffset_);
    dev_frag.odoffset_ = ArrayView<fid_t*>(d_odoffset_);
    dev_frag.iodoffset_ = ArrayView<fid_t*>(d_iodoffset_);

    dev_frag.iespliters_ = ArrayView<ArrayView<nbr_t*>>(d_iespliters_);
    dev_frag.oespliters_ = ArrayView<ArrayView<nbr_t*>>(d_oespliters_);

    dev_frag.outer_vertices_of_frag_ =
        ArrayView<vertex_range_t>(d_outer_vertices_of_frag_);
    dev_frag.mirrors_of_frag_ =
        ArrayView<ArrayView<vertex_t>>(d_mirrors_of_frag_);

    return dev_frag;
  }

  void __allocate_device_fragment__() {
    auto& comm_spec = vm_ptr_->GetCommSpec();
    auto& ie = ie_.get_edges();
    auto& ieoffset = ie_.get_offsets();
    auto& oe = oe_.get_edges();
    auto& oeoffset = oe_.get_offsets();

    int dev_id = comm_spec.local_id();
    CHECK_CUDA(cudaSetDevice(dev_id));
    Stream stream;

    d_vm_ptr_ = std::make_shared<dev_vertex_map_t>(vm_ptr_);
    auto offset_size = ivnum_ + ovnum_ + 1;
    auto compute_prefix_sum =
        [offset_size](
            const grape::Array<nbr_t*, grape::Allocator<nbr_t*>>& offset) {
          pinned_vector<VID_T> prefix_sum(offset_size);

          for (vid_t idx = 0; idx < offset_size; idx++) {
            prefix_sum[idx] = offset[idx] - offset[0];
          }
          return prefix_sum;
        };

    if (load_strategy == grape::LoadStrategy::kOnlyIn ||
        load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_ieoffset_.resize(offset_size);
      d_ie_.resize(ie_.edge_num());
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_ie_.data()),
                                 ie.data(), sizeof(nbr_t) * ie_.edge_num(),
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));

      auto prefix_sum = compute_prefix_sum(ieoffset);
      ArrayView<VID_T> d_prefix_sum(prefix_sum.data(), prefix_sum.size());

      CalculateOffsetWithPrefixSum<nbr_t, vid_t>(
          stream, d_prefix_sum, thrust::raw_pointer_cast(d_ie_.data()),
          thrust::raw_pointer_cast(d_ieoffset_.data()));
    }

    if (load_strategy == grape::LoadStrategy::kOnlyOut ||
        load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_oeoffset_.resize(offset_size);
      d_oe_.resize(oe_.edge_num());
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_oe_.data()),
                                 oe.data(), sizeof(nbr_t) * oe_.edge_num(),
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));

      auto prefix_sum = compute_prefix_sum(oeoffset);
      ArrayView<VID_T> d_prefix_sum(prefix_sum.data(), prefix_sum.size());

      CalculateOffsetWithPrefixSum<nbr_t, vid_t>(
          stream, d_prefix_sum, thrust::raw_pointer_cast(d_oe_.data()),
          thrust::raw_pointer_cast(d_oeoffset_.data()));
    }

    if (sizeof(internal_vertex_t) > sizeof(VID_T)) {
      d_vdata_.resize(ivnum_ + ovnum_);
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_vdata_.data()),
                                 vdata_.data(),
                                 sizeof(VDATA_T) * (ivnum_ + ovnum_),
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));
    }

    d_ovgid_.resize(ovnum_);
    CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_ovgid_.data()),
                               ovgid_.data(), sizeof(VID_T) * ovnum_,
                               cudaMemcpyHostToDevice, stream.cuda_stream()));
    {
      auto size = ovg2l_.size();
      pinned_vector<VID_T> gids(size);
      pinned_vector<VID_T> lids(size);
      size_t idx = 0;

      d_ovg2l_ = std::shared_ptr<CUDASTL::HashMap<VID_T, VID_T>>(
          CUDASTL::CreateHashMap<VID_T, VID_T, CUDASTL::HashFunc<VID_T>>(
              stream.cuda_stream(), ovg2l_.bucket_count(), size),
          [this, dev_id](CUDASTL::HashMap<VID_T, VID_T>* hash_map) {
            CHECK_CUDA(cudaSetDevice(dev_id));
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
      d_outer_vertices_of_frag_[fid] = outer_vertices_of_frag_[fid];
    }

    stream.Sync();

    VLOG(1) << "fid: " << fid_ << " ivnum: " << ivnum_ << " ovnum: " << ovnum_
            << " ienum: " << ie_.edge_num() << " oenum: " << oe_.edge_num();
  }

  void OffloadTopology() const {
    d_ie_.clear();
    d_ie_.shrink_to_fit();

    d_oe_.clear();
    d_oe_.shrink_to_fit();
  }

  void ReloadTopology() const {
    auto& ie = ie_.get_edges();
    auto& oe = oe_.get_edges();
    Stream stream;
    if (load_strategy == grape::LoadStrategy::kOnlyIn ||
        load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_ie_.resize(ie_.edge_num());
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_ie_.data()),
                                 ie.data(), sizeof(nbr_t) * ie_.edge_num(),
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));
    }

    if (load_strategy == grape::LoadStrategy::kOnlyOut ||
        load_strategy == grape::LoadStrategy::kBothOutIn) {
      d_oe_.resize(oe_.edge_num());
      CHECK_CUDA(cudaMemcpyAsync(thrust::raw_pointer_cast(d_oe_.data()),
                                 oe.data(), sizeof(nbr_t) * oe_.edge_num(),
                                 cudaMemcpyHostToDevice, stream.cuda_stream()));
    }
    stream.Sync();
  }

  void ReleaseDeviceCSR() {
    d_ie_.resize(0);
    d_ie_.shrink_to_fit();

    d_oe_.resize(0);
    d_oe_.shrink_to_fit();

    d_ieoffset_.resize(0);
    d_ieoffset_.shrink_to_fit();

    d_oeoffset_.resize(0);
    d_oeoffset_.shrink_to_fit();

    d_iespliters_holder_.resize(0);
    d_iespliters_holder_.shrink_to_fit();

    d_oespliters_holder_.resize(0);
    d_oespliters_holder_.shrink_to_fit();

    d_iespliters_.resize(0);
    d_iespliters_.shrink_to_fit();

    d_oespliters_.resize(0);
    d_oespliters_.shrink_to_fit();
  }

  std::shared_ptr<coo_t> ConvertToCOO(bool release_csr = false) {
    if (coo_frag_ == nullptr) {
      if (release_csr) {
        ReleaseDeviceCSR();
      }

      thrust::host_vector<typename coo_t::edge_t> edges;

      edges.reserve(GetEdgeNum());

      for (auto u : InnerVertices()) {
        auto oe = GetOutgoingAdjList(u);

        for (auto& e : oe) {
          auto v = e.get_neighbor();
          auto data = e.get_data();

          edges.push_back(
              typename coo_t::edge_t(u.GetValue(), v.GetValue(), data));
        }
      }

      coo_frag_ = std::make_shared<coo_t>();
      coo_frag_->Init(edges);
    }
    return coo_frag_;
  }

 public:
  // This is a restriction of the extended device lambda. From CUDA C
  // Programming guide:
  // If the enclosing function is a class member, then the following conditions
  // must be satisfied :
  //   - All classes enclosing the member function must have a name. The member
  //     function must not have private or
  //   - protected access within its parent class. All enclosing classes must
  //     not have private or
  //   - protected access within their respective parent classes.
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-lambda-restrictions
  void __init_edges_splitter__(
      const Stream& stream,
      grape::Array<nbr_t*, grape::Allocator<nbr_t*>> const& eoffset,
      std::vector<grape::VertexArray<inner_vertices_t, nbr_t*>> const&
          espliters,
      thrust::device_vector<nbr_t*>& d_eoffset,
      std::vector<thrust::device_vector<nbr_t*>>& d_espliters_holder,
      thrust::device_vector<ArrayView<nbr_t*>>& d_espliters) {
    if (!espliters.empty()) {
      d_espliters_holder.resize(1);
      d_espliters_holder[0].resize(ivnum_);
      d_espliters.push_back(ArrayView<nbr_t*>(d_espliters_holder[0]));
      pinned_vector<size_t> h_degree(espliters[0].size());
      int i = 0;
      for (auto v : InnerVertices()) {
        h_degree[i++] = espliters[0][v] - eoffset[0];
      }
      LaunchKernel(
          stream,
          [] __device__(size_t * h_degree, vid_t ivnum,
                        ArrayView<nbr_t*> offset, ArrayView<nbr_t*> espliter) {
            auto tid = TID_1D;
            auto nthreads = TOTAL_THREADS_1D;
            for (size_t i = 0 + tid; i < ivnum; i += nthreads) {
              espliter[i] = offset[0] + h_degree[i];
            }
          },
          thrust::raw_pointer_cast(h_degree.data()), ivnum_,
          ArrayView<nbr_t*>(d_eoffset), ArrayView<nbr_t*>(d_espliters[0]));
    }
  }

  void __init_edges_splitter_by_fragment__(
      const Stream& stream,
      grape::Array<nbr_t*, grape::Allocator<nbr_t*>> const& eoffset,
      std::vector<grape::VertexArray<inner_vertices_t, nbr_t*>> const&
          espliters,
      thrust::device_vector<nbr_t*>& d_eoffset,
      std::vector<thrust::device_vector<nbr_t*>>& d_espliters_holder,
      thrust::device_vector<ArrayView<nbr_t*>>& d_espliters) {
    d_espliters_holder.resize(fnum_ + 1);
    for (auto& vec : d_espliters_holder) {
      vec.resize(ivnum_);
      d_espliters.push_back(ArrayView<nbr_t*>(vec));
    }
    for (fid_t fid = 0; fid < fnum_ + 1; fid++) {
      auto& e_splitter = espliters[fid];

      if (!e_splitter.empty()) {
        pinned_vector<size_t> h_degree(e_splitter.size());
        int i = 0;
        for (auto v : InnerVertices()) {
          h_degree[i++] = e_splitter[v] - eoffset[0];
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
  void __initMessageDestination(const Stream& stream,
                                const grape::MessageStrategy& msg_strategy) {
    if (msg_strategy ==
        grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      __initDestFidList(stream, false, true, odst_.get_edges(),
                        odst_.get_offsets(), d_odst_, d_odoffset_);
    } else if (msg_strategy ==
               grape::MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      __initDestFidList(stream, true, false, idst_.get_edges(),
                        idst_.get_offsets(), d_idst_, d_idoffset_);
    } else if (msg_strategy ==
               grape::MessageStrategy::kAlongEdgeToOuterVertex) {
      __initDestFidList(stream, true, true, iodst_.get_edges(),
                        iodst_.get_offsets(), d_iodst_, d_iodoffset_);
    }
  }

  void __initDestFidList(
      const Stream& stream, bool in_edge, bool out_edge,
      grape::Array<fid_t, grape::Allocator<fid_t>> const& fid_list,
      grape::Array<fid_t*, grape::Allocator<fid_t*>> const& fid_list_offset,
      thrust::device_vector<fid_t>& d_fid_list,
      thrust::device_vector<fid_t*>& d_fid_list_offset) {
    pinned_vector<size_t> prefix_sum(ivnum_ + 1, 0);
    ArrayView<size_t> d_prefix_sum(prefix_sum.data(), prefix_sum.size());

    for (VID_T i = 0; i < ivnum_; ++i) {
      prefix_sum[i + 1] =
          prefix_sum[i] + (fid_list_offset[i + 1] - fid_list_offset[i]);
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

  void __initMirrorInfo(const CommSpec& comm_spec) {
    int dev_id = comm_spec.local_id();
    CHECK_CUDA(cudaSetDevice(dev_id));

    for (fid_t i = 0; i < fnum_; ++i) {
      d_mirrors_of_frag_holder_[i] = mirrors_of_frag_[i];
      d_mirrors_of_frag_[i] = ArrayView<vertex_t>(d_mirrors_of_frag_holder_[i]);
    }
  }

  using base_t::directed_;
  using base_t::fid_;
  using base_t::fnum_;
  using base_t::id_parser_;
  using base_t::ivnum_;
  using base_t::ovnum_;
  using base_t::vm_ptr_;

  using base_t::ovg2l_;
  using base_t::ovgid_;
  using base_t::vdata_;

  using base_t::mirrors_of_frag_;
  using base_t::outer_vertices_of_frag_;

  // CSR
  using base_t::ie_;
  using base_t::oe_;

  // CSR
  using base_t::idst_;
  using base_t::iodst_;
  using base_t::odst_;

  using base_t::iespliters_;
  using base_t::oespliters_;

  std::shared_ptr<dev_vertex_map_t> d_vm_ptr_;
  std::shared_ptr<CUDASTL::HashMap<VID_T, VID_T>> d_ovg2l_;
  thrust::device_vector<VID_T> d_ovgid_;
  mutable thrust::device_vector<nbr_t> d_ie_, d_oe_;

  thrust::device_vector<nbr_t*> d_ieoffset_, d_oeoffset_;
  thrust::device_vector<VDATA_T> d_vdata_;

  thrust::device_vector<fid_t> d_idst_, d_odst_, d_iodst_;
  thrust::device_vector<fid_t*> d_idoffset_, d_odoffset_, d_iodoffset_;

  std::vector<thrust::device_vector<nbr_t*>> d_iespliters_holder_,
      d_oespliters_holder_;
  thrust::device_vector<ArrayView<nbr_t*>> d_iespliters_, d_oespliters_;

  thrust::device_vector<vertex_range_t> d_outer_vertices_of_frag_;

  std::vector<thrust::device_vector<vertex_t>> d_mirrors_of_frag_holder_;
  thrust::device_vector<ArrayView<vertex_t>> d_mirrors_of_frag_;

  std::shared_ptr<coo_t> coo_frag_;
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_FRAGMENT_HOST_FRAGMENT_H_
