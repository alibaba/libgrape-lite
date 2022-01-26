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

#ifndef GRAPE_CUDA_FRAGMENT_COO_FRAGMENT_H_
#define GRAPE_CUDA_FRAGMENT_COO_FRAGMENT_H_

#include <cooperative_groups.h>

#include "grape/cuda/utils/dev_utils.h"
#include "grape/cuda/utils/shared_value.h"
#include "grape/cuda/utils/vertex_array.h"
#include "grape/types.h"

namespace grape {
namespace cuda {
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

  void Init(const thrust::host_vector<edge_t>& edges) { edges_ = edges; }

  device_t DeviceObject() { return device_t(ArrayView<edge_t>(edges_)); }

  size_t GetEdgeNum() const { return edges_.size(); }

 private:
  thrust::device_vector<edge_t> edges_;
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_FRAGMENT_COO_FRAGMENT_H_
