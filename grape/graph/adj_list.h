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

#ifndef GRAPE_GRAPH_ADJ_LIST_H_
#define GRAPE_GRAPH_ADJ_LIST_H_

#include <utility>

#include "grape/graph/edge.h"
#include "grape/utils/vertex_array.h"

namespace grape {

/**
 * @brief A neighbor of a vertex in the graph.
 *
 * Assume an edge, vertex_a --(edge_data)--> vertex_b.
 * a <Nbr> of vertex_a stores <Vertex> b and the edge_data.
 *
 * @tparam VID_T
 * @tparam EDATA_T
 */
template <typename VID_T, typename EDATA_T>
struct Nbr {
  Nbr() : neighbor(), data() {}
  explicit Nbr(const VID_T& nbr_) : neighbor(nbr_), data() {}
  explicit Nbr(const Vertex<VID_T>& nbr_) : neighbor(nbr_), data() {}
  Nbr(const Nbr& rhs) : neighbor(rhs.neighbor), data(rhs.data) {}
  Nbr(const VID_T& nbr_, const EDATA_T& data_) : neighbor(nbr_), data(data_) {}
  Nbr(const Vertex<VID_T>& nbr_, const EDATA_T& data_)
      : neighbor(nbr_), data(data_) {}
  ~Nbr() {}

  Nbr& operator=(const Nbr& rhs) {
    neighbor = rhs.neighbor;
    data = rhs.data;
    return *this;
  }

  void GetEdgeSrc(const Edge<VID_T, EDATA_T>& edge) {
    neighbor.SetValue(edge.src());
    data = edge.edata();
  }

  void GetEdgeDst(const Edge<VID_T, EDATA_T>& edge) {
    neighbor.SetValue(edge.dst());
    data = edge.edata();
  }

  Vertex<VID_T> get_neighbor() const { return neighbor; }
  EDATA_T get_data() const { return data; }

  Vertex<VID_T> neighbor;
  EDATA_T data;
};

/**
 * @brief A neighbor of a vertex in the graph. (partial specialization with
 * Empty edge data.)
 *
 * @tparam VID_T
 */
template <typename VID_T>
struct Nbr<VID_T, EmptyType> {
  Nbr() : neighbor() {}
  explicit Nbr(const VID_T& nbr_) : neighbor(nbr_) {}
  explicit Nbr(const Vertex<VID_T>& nbr_) : neighbor(nbr_) {}
  Nbr(const Nbr& rhs) : neighbor(rhs.neighbor) {}
  Nbr(const VID_T& nbr_, const EmptyType&) : neighbor(nbr_) {}
  Nbr(const Vertex<VID_T>& nbr_, const EmptyType&) : neighbor(nbr_) {}
  ~Nbr() {}

  Nbr& operator=(const Nbr& rhs) {
    neighbor = rhs.neighbor;
    return *this;
  }

  void GetEdgeSrc(const Edge<VID_T, EmptyType>& edge) {
    neighbor.SetValue(edge.src());
  }

  void GetEdgeDst(const Edge<VID_T, EmptyType>& edge) {
    neighbor.SetValue(edge.dst());
  }

  Vertex<VID_T> get_neighbor() const { return neighbor; }
  EmptyType get_data() const { return data; }

  union {
    Vertex<VID_T> neighbor;
    EmptyType data;
  };
};

template <typename VID_T, typename EDATA_T>
inline InArchive& operator<<(InArchive& archive,
                             const Nbr<VID_T, EDATA_T>& nbr) {
  archive << nbr.neighbor << nbr.data;
  return archive;
}

template <typename VID_T>
inline InArchive& operator<<(InArchive& archive,
                             const Nbr<VID_T, EmptyType>& nbr) {
  archive << nbr.neighbor;
  return archive;
}

template <typename VID_T, typename EDATA_T>
inline OutArchive& operator>>(OutArchive& archive, Nbr<VID_T, EDATA_T>& nbr) {
  archive >> nbr.neighbor >> nbr.data;
  return archive;
}

template <typename VID_T>
inline OutArchive& operator>>(OutArchive& archive, Nbr<VID_T, EmptyType>& nbr) {
  archive >> nbr.neighbor;
  return archive;
}

/**
 * @brief A iteratable adjencent list of a vertex. The list contains all
 * neighbors in format of Nbr, which contains the other Node and the data on the
 * Edge.
 *
 * @tparam VID_T
 * @tparam EDATA_T
 */
template <typename VID_T, typename EDATA_T>
class AdjList {
  using NbrT = Nbr<VID_T, EDATA_T>;

 public:
  AdjList() {}
  AdjList(NbrT* b, NbrT* e) : begin_(b), end_(e) {}
  ~AdjList() {}

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return !Empty(); }

  inline size_t Size() const { return end_ - begin_; }

  class iterator {
    using pointer_type = NbrT*;
    using reference_type = NbrT&;

   private:
    NbrT* current_;

   public:
    iterator() noexcept : current_() {}
    explicit iterator(const pointer_type& c) noexcept : current_(c) {}
    reference_type operator*() const noexcept { return *current_; }
    pointer_type operator->() const noexcept { return current_; }

    iterator& operator++() noexcept {
      ++current_;
      return *this;
    }

    iterator operator++(int) noexcept { return iterator(current_++); }

    iterator& operator--() noexcept {
      --current_;
      return *this;
    }

    iterator operator--(int) noexcept { return iterator(current_--); }

    iterator operator+(size_t offset) noexcept {
      return iterator(current_ + offset);
    }

    bool operator==(const iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    bool operator!=(const iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  class const_iterator {
    using pointer_type = const NbrT*;
    using reference_type = const NbrT&;

   private:
    const NbrT* current_;

   public:
    const_iterator() noexcept : current_() {}
    explicit const_iterator(const pointer_type& c) noexcept : current_(c) {}
    reference_type operator*() const noexcept { return *current_; }
    pointer_type operator->() const noexcept { return current_; }

    const_iterator& operator++() noexcept {
      ++current_;
      return *this;
    }

    const_iterator operator++(int) noexcept {
      return const_iterator(current_++);
    }

    const_iterator& operator--() noexcept {
      --current_;
      return *this;
    }

    const_iterator operator--(int) noexcept {
      return const_iterator(current_--);
    }

    const_iterator operator+(size_t offset) noexcept {
      return const_iterator(current_ + offset);
    }

    bool operator==(const const_iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    bool operator!=(const const_iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  iterator begin() { return iterator(begin_); }
  iterator end() { return iterator(end_); }

  const_iterator begin() const { return const_iterator(begin_); }
  const_iterator end() const { return const_iterator(end_); }

  NbrT* begin_pointer() { return begin_; }
  const NbrT* begin_pointer() const { return begin_; }

  NbrT* end_pointer() { return end_; }
  const NbrT* end_pointer() const { return end_; }

  bool empty() const { return end_ == begin_; }

 private:
  NbrT* begin_;
  NbrT* end_;
};

/**
 * @brief A immutable iteratable adjencent list of a vertex. The list contains
 * all neighbors in format of Nbr, which contains the other Node and the data on
 * the Edge.
 *
 * @tparam VID_T
 * @tparam EDATA_T
 */
template <typename VID_T, typename EDATA_T>
class ConstAdjList {
  using NbrT = Nbr<VID_T, EDATA_T>;

 public:
  ConstAdjList() {}
  ConstAdjList(NbrT* b, NbrT* e) : begin_(b), end_(e) {}
  ~ConstAdjList() {}

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return !Empty(); }

  inline size_t Size() const { return end_ - begin_; }

  class const_iterator {
    using pointer_type = const NbrT*;
    using reference_type = const NbrT&;

   private:
    const NbrT* current_;

   public:
    const_iterator() noexcept : current_() {}
    explicit const_iterator(const pointer_type& c) noexcept : current_(c) {}
    reference_type operator*() const noexcept { return *current_; }
    pointer_type operator->() const noexcept { return current_; }

    const_iterator& operator++() noexcept {
      ++current_;
      return *this;
    }

    const_iterator operator++(int) noexcept {
      return const_iterator(current_++);
    }

    const_iterator& operator--() noexcept {
      --current_;
      return *this;
    }

    const_iterator operator--(int) noexcept {
      return const_iterator(current_--);
    }

    const_iterator operator+(size_t offset) noexcept {
      return const_iterator(current_ + offset);
    }

    bool operator==(const const_iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    bool operator!=(const const_iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  const_iterator begin() const { return const_iterator(begin_); }
  const_iterator end() const { return const_iterator(end_); }

  const NbrT* begin_pointer() const { return begin_; }

  const NbrT* end_pointer() const { return end_; }

  bool empty() const { return end_ == begin_; }

 private:
  const NbrT* begin_;
  const NbrT* end_;
};

/**
 * @brief Destination list for message exchange. A message may need to be sent
 * to all the fragments in the DestList.
 *
 */
struct DestList {
  DestList(fid_t* begin_, fid_t* end_) : begin(begin_), end(end_) {}
  inline bool Empty() { return begin == end; }
  inline bool NotEmpty() { return !Empty(); }
  fid_t* begin;
  fid_t* end;
};

}  // namespace grape

#endif  // GRAPE_GRAPH_ADJ_LIST_H_
