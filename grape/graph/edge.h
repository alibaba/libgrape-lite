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

#ifndef GRAPE_GRAPH_EDGE_H_
#define GRAPE_GRAPH_EDGE_H_

#include <utility>

#include "grape/types.h"

namespace grape {

class InArchive;
class OutArchive;

/**
 * @brief Edge representation.
 *
 * @tparam VID_T
 * @tparam EDATA_T
 */
template <typename VID_T, typename EDATA_T>
class Edge {
 public:
  Edge() : src_(), dst_(), edata_() {}
  ~Edge() {}

  Edge(const VID_T& src, const VID_T& dst) : src_(src), dst_(dst), edata_() {}
  Edge(const VID_T& src, const VID_T& dst, const EDATA_T& edata)
      : src_(src), dst_(dst), edata_(edata) {}
  Edge(const Edge& e) : src_(e.src_), dst_(e.dst_), edata_(e.edata_) {}

  inline const VID_T& src() const { return src_; }
  inline const VID_T& dst() const { return dst_; }
  inline const EDATA_T& edata() const { return edata_; }

  void SetEndpoint(const VID_T& src, const VID_T& dst) {
    src_ = src;
    dst_ = dst;
  }

  void set_src(const VID_T& src) { src_ = src; }

  void set_dst(const VID_T& dst) { dst_ = dst; }

  void set_edata(const EDATA_T& edata) { edata_ = edata; }

  void set_edata(EDATA_T&& edata) { edata_ = std::move(edata); }

  Edge& operator=(const Edge& other) {
    src_ = other.src();
    dst_ = other.dst();
    edata_ = other.edata();
    return *this;
  }

  bool operator==(const Edge& other) const {
    return src_ == other.src() && dst_ == other.dst();
  }

  bool operator!=(const Edge& other) const { return !(*this == other); }

 private:
  VID_T src_;
  VID_T dst_;
  EDATA_T edata_;

  template <typename _OID_T, typename _VID_T, typename _VDATA_T,
            typename _EDATA_T, LoadStrategy _load_strategy>
  friend class ImmutableEdgecutFragment;

  template <typename _FRAG_T, typename _PARTITIONER_T, typename _IOADAPTOR_T,
            typename _Enable>
  friend class BasicFragmentLoader;

  template <typename FRAG_T,
          typename PARTITIONER_T,
          typename IOADAPTOR_T,
          typename LINE_PARSER_T>
  friend class EVFragmentLoader;

  template <typename _FRAG_T, typename _Enable>
  friend class Rebalancer;

  friend InArchive& operator<<(InArchive& archive,
                               const Edge<VID_T, EDATA_T>& e) {
    archive << e.src_ << e.dst_ << e.edata_;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive, Edge<VID_T, EDATA_T>& e) {
    archive >> e.src_ >> e.dst_ >> e.edata_;
    return archive;
  }
};

/**
 * @brief Partial specialization for Edge with EmptyType on edge_data.
 *
 * @tparam VID_T
 */
template <typename VID_T>
class Edge<VID_T, EmptyType> {
 public:
  Edge() : src_(), dst_() {}
  Edge(const VID_T& src, const VID_T& dst) : src_(src), dst_(dst) {}
  Edge(const VID_T& src, const VID_T& dst, const EmptyType& edata)
      : src_(src), dst_(dst) {}
  Edge(const Edge& e) : src_(e.src_), dst_(e.dst_) {}
  ~Edge() {}

  inline const VID_T& src() const { return src_; }
  inline const VID_T& dst() const { return dst_; }
  inline const EmptyType& edata() const { return edata_; }

  void SetEndpoint(const VID_T& src, const VID_T& dst) {
    src_ = src;
    dst_ = dst;
  }

  void set_edata(const EmptyType& edata) {}

  Edge& operator=(const Edge& other) {
    src_ = other.src();
    dst_ = other.dst();
    return *this;
  }

  bool operator==(const Edge& other) const {
    return src_ == other.src() && dst_ == other.dst();
  }

  bool operator!=(const Edge& other) const { return !(*this == other); }

 private:
  VID_T src_;
  union {
    VID_T dst_;
    EmptyType edata_;
  };

  template <typename _OID_T, typename _VID_T, typename _VDATA_T,
            typename _EDATA_T, LoadStrategy _load_strategy>
  friend class ImmutableEdgecutFragment;

  template <typename _FRAG_T, typename _PARTITIONER_T, typename _IOADAPTOR_T,
            typename _Enable>
  friend class BasicFragmentLoader;

  template <typename FRAG_T,
          typename PARTITIONER_T,
          typename IOADAPTOR_T,
          typename LINE_PARSER_T>
  friend class EVFragmentLoader;

  template <typename _FRAG_T, typename _Enable>
  friend class Rebalancer;

  friend InArchive& operator<<(InArchive& archive,
                               const Edge<VID_T, EmptyType>& e) {
    archive << e.src_ << e.dst_;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive,
                                Edge<VID_T, EmptyType>& e) {
    archive >> e.src_ >> e.dst_;
    return archive;
  }
};

}  // namespace grape

#endif  // GRAPE_GRAPH_EDGE_H_
