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

#ifndef GRAPE_GRAPH_VERTEX_H_
#define GRAPE_GRAPH_VERTEX_H_

#include <utility>

#include "grape/types.h"

namespace grape {

namespace internal {

class InArchive;
class OutArchive;

template <typename FRAG_T, typename PARTITIONER_T, typename IOADAPTOR_T,
          typename Enable>
class BasicFragmentLoader;

/**
 * @brief Vertex representation.
 *
 * @tparam VID_T
 * @tparam VDATA_T
 */
template <typename VID_T, typename VDATA_T>
class Vertex {
 public:
  DEV_HOST Vertex() = default;

  DEV_HOST explicit Vertex(const VID_T& vid) : vid_(vid), vdata_() {}
  DEV_HOST Vertex(const VID_T& vid, const VDATA_T& vdata) : vid_(vid), vdata_(vdata) {}
  DEV_HOST Vertex(const Vertex& vert) : vid_(vert.vid_), vdata_(vert.vdata_) {}

  DEV_HOST ~Vertex() {}

  DEV_HOST Vertex& operator=(const Vertex& rhs) {
    if (this == &rhs) {
      return *this;
    }
    vid_ = rhs.vid_;
    vdata_ = rhs.vdata_;
    return *this;
  }

  DEV_HOST_INLINE const VID_T& vid() const { return vid_; }
  DEV_HOST_INLINE const VDATA_T& vdata() const { return vdata_; }

  DEV_HOST_INLINE void set_vid(const VID_T& vid) { vid_ = vid; }
  DEV_HOST_INLINE void set_vdata(const VDATA_T& vdata) { vdata_ = vdata; }
  DEV_HOST_INLINE void set_vdata(VDATA_T&& vdata) { vdata_ = std::move(vdata); }

 private:
  template <typename _FRAG_T, typename _PARTITIONER_T, typename _IOADAPTOR_T,
            typename _Enable>
  friend class BasicFragmentLoader;
  VID_T vid_;
  VDATA_T vdata_;

  friend InArchive& operator<<(InArchive& archive,
                               const Vertex<VID_T, VDATA_T>& v) {
    archive << v.vid_;
    archive << v.vdata_;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive,
                                Vertex<VID_T, VDATA_T>& v) {
    archive >> v.vid_;
    archive >> v.vdata_;
    return archive;
  }
};

/**
 * @brief Partial specialization for Vertex with EmptyType on vertex_data.
 *
 */
template <typename VID_T>
class Vertex<VID_T, EmptyType> {
 public:
  Vertex() : vid_() {}

  DEV_HOST explicit Vertex(const VID_T& vid) : vid_(vid) {}
  DEV_HOST Vertex(const VID_T& vid, const EmptyType&) : vid_(vid) {}
  DEV_HOST Vertex(const Vertex& vert) : vid_(vert.vid_) {}

  DEV_HOST ~Vertex() {}

  DEV_HOST Vertex& operator=(const Vertex& rhs) {
    if (this == &rhs) {
      return *this;
    }
    vid_ = rhs.vid_;
    return *this;
  }

  DEV_HOST_INLINE const VID_T& vid() const { return vid_; }
  DEV_HOST_INLINE const EmptyType& vdata() const { return vdata_; }

  DEV_HOST_INLINE void set_vid(const VID_T& vid) { vid_ = vid; }
  DEV_HOST_INLINE void set_vdata(const EmptyType& vdata) {}

 private:
  union {
    VID_T vid_;
    EmptyType vdata_;
  };
  template <typename _FRAG_T, typename _PARTITIONER_T, typename _IOADAPTOR_T,
            typename _Enable>
  friend class BasicFragmentLoader;

  friend InArchive& operator<<(InArchive& archive,
                               const Vertex<VID_T, EmptyType>& v) {
    archive << v.vid_;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive,
                                Vertex<VID_T, EmptyType>& v) {
    archive >> v.vid_;
    return archive;
  }
};

}  // namespace internal

}  // namespace grape

#endif  // GRAPE_GRAPH_VERTEX_H_
