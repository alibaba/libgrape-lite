#ifndef GRAPE_GPU_FRAGMENT_COO_FRAGMENT_H_
#define GRAPE_GPU_FRAGMENT_COO_FRAGMENT_H_
#include <cooperative_groups.h>

#include <unordered_set>

#include "grape/types.h"
#include "grape/vertex_map/global_vertex_map.h"
#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/vertex_array.h"

namespace grape_gpu {

enum class COOIdType { kConsecutive, kLocal, kOID };

template <typename VID_T, typename EDATA_T>
class Edge {
 public:
  Edge() = default;

  DEV_HOST Edge(Vertex<VID_T> src, Vertex<VID_T> dst, EDATA_T data)
      : src_(src), dst_(dst), data_(data) {}

  DEV_HOST_INLINE Vertex<VID_T> src() const { return src_; }

  DEV_HOST_INLINE Vertex<VID_T> dst() const { return dst_; }

  DEV_HOST_INLINE EDATA_T& data() { return data_; }

  DEV_HOST_INLINE const EDATA_T& data() const { return data_; }

 private:
  Vertex<VID_T> src_;
  Vertex<VID_T> dst_;
  EDATA_T data_;
};

template <typename VID_T>
class Edge<VID_T, grape::EmptyType> {
 public:
  Edge() = default;

  DEV_HOST Edge(const Edge<VID_T, grape::EmptyType>& rhs) {
    src_ = rhs.src_;
    dst_ = rhs.dst_;
  }

  DEV_HOST_INLINE Edge<VID_T, grape::EmptyType>& operator=(
      const Edge<VID_T, grape::EmptyType>& rhs) {
    src_ = rhs.src_;
    dst_ = rhs.dst_;
    return *this;
  }

  DEV_HOST Edge(Vertex<VID_T> src, Vertex<VID_T> dst) : src_(src), dst_(dst) {}

  DEV_HOST Edge(Vertex<VID_T> src, Vertex<VID_T> dst, grape::EmptyType)
      : src_(src), dst_(dst) {}

  DEV_HOST_INLINE Vertex<VID_T> src() const { return src_; }

  DEV_HOST_INLINE Vertex<VID_T> dst() const { return dst_; }

  DEV_HOST_INLINE grape::EmptyType& data() { return data_; }

  DEV_HOST_INLINE const grape::EmptyType& data() const { return data_; }

 private:
  Vertex<VID_T> src_;
  union {
    Vertex<VID_T> dst_;
    grape::EmptyType data_;
  };
};

namespace dev {
template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class COOFragment {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = Vertex<vid_t>;
  using edge_t = Edge<vid_t, EDATA_T>;
  using vertex_map_t = grape::GlobalVertexMap<oid_t, vid_t>;

  COOFragment() = default;

  DEV_HOST COOFragment(ArrayView<edge_t> edges) : edges_(edges) {}

  DEV_INLINE const edge_t& edge(size_t eid) const {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_INLINE edge_t& edge(size_t eid) {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_INLINE edge_t& operator[](size_t eid) const {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_INLINE edge_t& operator[](size_t eid) {
    assert(eid < edges_.size());
    return edges_[eid];
  }

  DEV_HOST_INLINE size_t GetEdgeNum() const { return edges_.size(); }

 private:
  ArrayView<edge_t> edges_;
};
}  // namespace dev

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class COOFragment {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using vdata_t = VDATA_T;
  using edata_t = EDATA_T;
  using vertex_t = Vertex<VID_T>;
  using edge_t = Edge<VID_T, EDATA_T>;
  using device_t = dev::COOFragment<OID_T, VID_T, VDATA_T, EDATA_T>;
  using vertex_map_t = grape::GlobalVertexMap<oid_t, vid_t>;

  COOFragment(std::shared_ptr<vertex_map_t> vm_ptr, COOIdType id_type)
      : vm_ptr_(vm_ptr), id_type_(id_type), fnum_(vm_ptr->GetFragmentNum()) {
    VID_T offset = 0;

    for (fid_t fid = 0; fid < fnum_; fid++) {
      auto nv = vm_ptr_->GetInnerVertexSize(fid);
      offsets_.push_back(offset);
      offset += nv;
    }
    offsets_.push_back(offset);
    id_parser_.Init(vm_ptr->GetFragmentNum());
  }

  template <typename FRAG_T>
  void Init(const FRAG_T& frag) {
    Stream stream;
    auto d_frag = frag.DeviceObject();
    auto vertices = frag.Vertices();
    auto ov = frag.OuterVertices();
    auto iv = frag.InnerVertices();

    auto total_nv = frag.GetTotalVerticesNum();
    auto vertex2consecutive_gid = [this, &frag](vertex_t v) {
      auto gid = frag.Vertex2Gid(v);
      auto fid = id_parser_.GetFid(gid);
      auto lid = id_parser_.GetLid(gid);

      return offsets_[fid] + lid;
    };

    int th_num = round_up(std::thread::hardware_concurrency(), frag.fnum());
    std::vector<std::thread> ths;
    std::mutex mutex;
    int device;

    CHECK_CUDA(cudaGetDevice(&device));

    edges_.reserve(frag.oenum_);

    for (int i = 0; i < th_num; i++) {
      ths.push_back(std::thread(
          [&](int tid) {
            auto chunk_size = round_up(iv.size(), th_num);
            auto begin = std::min(tid * chunk_size, iv.size());
            auto end = std::min((tid + 1) * chunk_size, iv.size());
            thrust::host_vector<edge_t> h_edges;

            CHECK_CUDA(cudaSetDevice(device));

            for (auto lid = begin; lid < end; lid++) {
              vertex_t u(lid);

              auto u_consecutive_gid = vertex2consecutive_gid(u);
              auto u_oid = frag.GetId(u);
              auto oes = frag.GetOutgoingAdjList(u);

              for (auto nbr : oes) {
                auto v = nbr.get_neighbor();
                auto data = nbr.get_data();
                edge_t e;

                switch (id_type_) {
                case COOIdType::kLocal: {
                  e = edge_t(u, v, data);
                  break;
                }
                case COOIdType::kConsecutive: {
                  e = edge_t(vertex_t(u_consecutive_gid),
                             vertex_t(vertex2consecutive_gid(v)), data);
                  break;
                }
                case COOIdType::kOID: {
                  auto v_oid = frag.GetId(v);
                  CHECK(u_oid >= 0 && v_oid >= 0)
                      << "edge out of bound: (" << u_oid << " " << v_oid << ")";
                  e = edge_t(vertex_t(u_oid), vertex_t(v_oid), data);
                  break;
                }
                }

                h_edges.push_back(e);
              }
            }

            std::lock_guard<std::mutex> lock(mutex);
            edges_.template insert(edges_.end(), h_edges.begin(),
                                   h_edges.end());
          },
          i));
    }

    std::vector<vertex_t> processed_vertices(vertices.size());

    for (size_t i = 0; i < vertices.size(); i++) {
      vertex_t v(i);
      switch (id_type_) {
      case COOIdType::kLocal: {
        processed_vertices[i] = v;
        break;
      }
      case COOIdType::kConsecutive: {
        processed_vertices[i] = vertex_t(vertex2consecutive_gid(v));
        break;
      }
      case COOIdType::kOID: {
        processed_vertices[i] = vertex_t(frag.GetId(v));
        break;
      }
      }
    }

    vertices_ = processed_vertices;

    for (auto& th : ths) {
      th.join();
    }
  }

  VertexRange<vid_t> InnerVerticesWithConsecutiveGid() {
    return VertexRange<vid_t>(offsets_[fid_], offsets_[fid_ + 1]);
  }

  device_t DeviceObject() { return device_t(ArrayView<edge_t>(edges_)); }

  size_t GetEdgeNum() const { return edges_.size(); }

  vid_t ConsecutiveGid2Gid(vid_t consecutive_gid) {
    fid_t fid = std::lower_bound(offsets_.begin(), offsets_.end(),
                                 consecutive_gid + 1) -
                offsets_.begin() - 1;
    vid_t begin_offset = offsets_[fid];
    vid_t lid = consecutive_gid - begin_offset;

    return id_parser_.Lid2Gid(fid, lid);
  }

  WorkSourceArray<vertex_t> Vertices() {
    return WorkSourceArray<vertex_t>(thrust::raw_pointer_cast(vertices_.data()),
                                     vertices_.size());
  }

 private:
  std::shared_ptr<vertex_map_t> vm_ptr_;
  COOIdType id_type_;
  fid_t fnum_;
  fid_t fid_;

  IdParser<VID_T> id_parser_;
  std::vector<vid_t> offsets_;  // prefix sum of inner vertices num
  thrust::device_vector<edge_t> edges_;
  thrust::device_vector<vertex_t> vertices_;
};
}  // namespace grape_gpu
#endif  // GRAPE_GPU_FRAGMENT_COO_FRAGMENT_H_
