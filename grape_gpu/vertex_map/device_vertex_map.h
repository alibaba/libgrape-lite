
#ifndef GRAPE_GPU_VERTEX_MAP_DEVICE_VERTEX_MAP_H_
#define GRAPE_GPU_VERTEX_MAP_DEVICE_VERTEX_MAP_H_
#include "cuda_hashmap/hash_map.h"
#include "grape/vertex_map/global_vertex_map.h"
#include "grape_gpu/fragment/id_parser.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/launcher.h"
#include "grape_gpu/utils/stream.h"

namespace grape_gpu {
template <typename OID_T, typename VID_T>
class DeviceVertexMap;

namespace dev {
template <typename OID_T, typename VID_T>
class DeviceVertexMap {
 public:
  DEV_INLINE bool GetOid(const VID_T& gid, OID_T& oid) const {
    auto fid = id_parser_.GetFid(gid);
    auto lid = id_parser_.GetLid(gid);
    return GetOid(fid, lid, oid);
  }

  DEV_INLINE bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) const {
    if (l2o_.data() == nullptr) {
      return false;
    }
    oid = l2o_[fid][lid];
    return true;
  }

  DEV_INLINE bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) const {
    if (o2l_.data() == nullptr) {
      return false;
    }
    auto* iter = o2l_[fid]->find(oid);

    if (iter == NULL) {
      return false;
    } else {
      auto lid = iter->value;
      gid = id_parser_.Lid2Gid(fid, lid);
      return true;
    }
  }

  DEV_INLINE bool GetGid(const OID_T& oid, VID_T& gid) const {
    for (fid_t fid = 0; fid < fnum_; fid++) {
      if (GetGid(fid, oid, gid)) {
        return true;
      }
    }
    return false;
  }

  IdParser<VID_T>& id_parser() { return id_parser_; }

 private:
  fid_t fnum_{};
  IdParser<VID_T> id_parser_;
  ArrayView<CUDASTL::HashMap<OID_T, VID_T>*> o2l_;
  ArrayView<ArrayView<OID_T>> l2o_;

  template <typename _OID_T, typename _VID_T>
  friend class grape_gpu::DeviceVertexMap;
};
}  // namespace dev

/**
 * @brief a kind of VertexMapBase which holds global mapping information in
 * each worker.
 *
 * @tparam OID_T
 * @tparam VID_T
 */
template <typename OID_T, typename VID_T>
class DeviceVertexMap {
 public:
  explicit DeviceVertexMap(
      std::shared_ptr<grape::GlobalVertexMap<OID_T, VID_T>> vm_ptr)
      : vm_ptr_(std::move(vm_ptr)) {}

  void Init(const Stream& stream) {
    auto& comm_spec = vm_ptr_->GetCommSpec();
    fid_t fnum = comm_spec.fnum();

    id_parser_.Init(fnum);
    d_o2l_.resize(fnum);
    d_l2o_.resize(fnum);
    d_l2o_ptr_.resize(fnum);

    for (fid_t fid = 0; fid < fnum; fid++) {
      auto ivnum = vm_ptr_->GetInnerVertexSize(fid);
      // TODO: replace this
      d_o2l_[fid] =
          CUDASTL::CreateHashMap<OID_T, VID_T, CUDASTL::HashFunc<OID_T>>(
              stream.cuda_stream(), ivnum / 10 + 1, ivnum);

      pinned_vector<OID_T> oids(ivnum);

      for (size_t lid = 0; lid < ivnum; lid++) {
        OID_T oid;
        CHECK(vm_ptr_->GetOid(fid, lid, oid));
        oids[lid] = oid;
      }

      LaunchKernel(
          stream,
          [] __device__(OID_T * oids, VID_T size,
                        CUDASTL::HashMap<OID_T, VID_T> * o2l) {
            auto tid = TID_1D;
            auto nthreads = TOTAL_THREADS_1D;

            for (VID_T lid = 0 + tid; lid < size; lid += nthreads) {
              OID_T oid = oids[lid];

              (*o2l)[oid] = lid;
            }
          },
          oids.data(), ivnum, d_o2l_[fid]);
      d_l2o_[fid].assign(oids.begin(), oids.end());
      d_l2o_ptr_[fid] = ArrayView<OID_T>(d_l2o_[fid]);
    }
  }

  dev::DeviceVertexMap<OID_T, VID_T> DeviceObject() {
    auto& comm_spec = vm_ptr_->GetCommSpec();
    dev::DeviceVertexMap<OID_T, VID_T> dev_vm;

    dev_vm.fnum_ = comm_spec.fnum();
    dev_vm.id_parser_ = id_parser_;

    // if device vm is built
    if (!d_o2l_.empty()) {
      dev_vm.o2l_ = ArrayView<CUDASTL::HashMap<OID_T, VID_T>*>(d_o2l_);
      dev_vm.l2o_ = ArrayView<ArrayView<OID_T>>(d_l2o_ptr_);
    }
    return dev_vm;
  }

 private:
  std::shared_ptr<grape::GlobalVertexMap<OID_T, VID_T>> vm_ptr_;
  IdParser<VID_T> id_parser_;
  // l2o for per device
  thrust::device_vector<
      CUDASTL::HashMap<OID_T, VID_T, CUDASTL::HashFunc<OID_T>>*>
      d_o2l_;
  std::vector<thrust::device_vector<OID_T>> d_l2o_;
  thrust::device_vector<ArrayView<OID_T>> d_l2o_ptr_;
};

}  // namespace grape_gpu
#endif  // GRAPE_GPU_VERTEX_MAP_DEVICE_VERTEX_MAP_H_
