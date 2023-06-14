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

#include "grape/config.h"
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
struct Edge {
  DEV_HOST Edge() : src(), dst(), edata() {}
  DEV_HOST ~Edge() {}

  DEV_HOST Edge(const VID_T& src, const VID_T& dst)
      : src(src), dst(dst), edata() {}
  DEV_HOST Edge(const VID_T& src, const VID_T& dst, const EDATA_T& edata)
      : src(src), dst(dst), edata(edata) {}
  DEV_HOST Edge(const Edge& e) : src(e.src), dst(e.dst), edata(e.edata) {}
  DEV_HOST Edge(const VID_T& src, const VID_T& dst, EDATA_T&& edata)
      : src(src), dst(dst), edata(std::move(edata)) {}
  DEV_HOST Edge(Edge&& e) noexcept
      : src(e.src), dst(e.dst), edata(std::move(e.edata)) {}

  DEV_HOST Edge& operator=(const Edge& other) {
    if (this == &other) {
      return *this;
    }
    src = other.src;
    dst = other.dst;
    edata = other.edata;
    return *this;
  }

  DEV_HOST Edge& operator=(Edge&& other) {
    if (this == &other) {
      return *this;
    }
    src = other.src;
    dst = other.dst;
    edata = std::move(other.edata);
    return *this;
  }

  DEV_HOST bool operator==(const Edge& other) const {
    return src == other.src && dst == other.dst;
  }

  DEV_HOST bool operator!=(const Edge& other) const {
    return !(*this == other);
  }

  VID_T src;
  VID_T dst;
  EDATA_T edata;
};

/**
 * @brief Partial specialization for Edge with EmptyType on edge_data.
 *
 * @tparam VID_T
 */
template <typename VID_T>
struct Edge<VID_T, EmptyType> {
  DEV_HOST Edge() : src(), dst() {}
  DEV_HOST Edge(const VID_T& src, const VID_T& dst) : src(src), dst(dst) {}
  DEV_HOST Edge(const VID_T& src, const VID_T& dst, const EmptyType& edata)
      : src(src), dst(dst) {}
  DEV_HOST Edge(const Edge& e) : src(e.src), dst(e.dst) {}
  DEV_HOST ~Edge() {}

  DEV_HOST Edge& operator=(const Edge& other) {
    if (this == &other) {
      return *this;
    }
    src = other.src;
    dst = other.dst;
    return *this;
  }

  DEV_HOST bool operator==(const Edge& other) const {
    return src == other.src && dst == other.dst;
  }

  DEV_HOST bool operator!=(const Edge& other) const {
    return !(*this == other);
  }

  VID_T src;
  union {
    VID_T dst;
    EmptyType edata;
  };
};

template <typename VID_T, typename EDATA_T>
InArchive& operator<<(InArchive& archive, const Edge<VID_T, EDATA_T>& e) {
  archive << e.src << e.dst << e.edata;
  return archive;
}

template <typename VID_T, typename EDATA_T>
OutArchive& operator>>(OutArchive& archive, Edge<VID_T, EDATA_T>& e) {
  archive >> e.src >> e.dst >> e.edata;
  return archive;
}

template <typename VID_T>
InArchive& operator<<(InArchive& archive, const Edge<VID_T, EmptyType>& e) {
  archive << e.src << e.dst;
  return archive;
}

template <typename VID_T>
OutArchive& operator>>(OutArchive& archive, Edge<VID_T, EmptyType>& e) {
  archive >> e.src >> e.dst;
  return archive;
}

}  // namespace grape

#endif  // GRAPE_GRAPH_EDGE_H_
