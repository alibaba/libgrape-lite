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

#include "grape/config.h"
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
  DEV_HOST Nbr() : neighbor(), data() {}
  DEV_HOST explicit Nbr(const VID_T& nbr_) : neighbor(nbr_), data() {}
  DEV_HOST explicit Nbr(const Vertex<VID_T>& nbr_) : neighbor(nbr_), data() {}
  DEV_HOST Nbr(const Nbr& rhs) : neighbor(rhs.neighbor), data(rhs.data) {}
  DEV_HOST Nbr(Nbr&& rhs) noexcept
      : neighbor(rhs.neighbor), data(std::move(rhs.data)) {}
  DEV_HOST Nbr(const VID_T& nbr_, const EDATA_T& data_)
      : neighbor(nbr_), data(data_) {}
  DEV_HOST Nbr(const Vertex<VID_T>& nbr_, const EDATA_T& data_)
      : neighbor(nbr_), data(data_) {}
  DEV_HOST Nbr(const VID_T& nbr_, EDATA_T&& data_)
      : neighbor(nbr_), data(std::move(data_)) {}
  DEV_HOST ~Nbr() {}

  DEV_HOST_INLINE Nbr& operator=(const Nbr& rhs) {
    neighbor = rhs.neighbor;
    data = rhs.data;
    return *this;
  }
  DEV_HOST_INLINE Nbr& operator=(Nbr&& rhs) {
    neighbor = std::move(rhs.neighbor);
    data = std::move(rhs.data);
    return *this;
  }

  DEV_HOST_INLINE Vertex<VID_T> get_neighbor() const { return neighbor; }
  DEV_HOST_INLINE const EDATA_T& get_data() const { return data; }

  DEV_HOST_INLINE VID_T get_neighbor_lid() const { return neighbor.GetValue(); }

  Vertex<VID_T> neighbor;
  EDATA_T data;
};

template <typename VID_T, typename EDATA_T>
bool operator<(const Nbr<VID_T, EDATA_T>& lhs, const Nbr<VID_T, EDATA_T>& rhs) {
  return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
}

/**
 * @brief A neighbor of a vertex in the graph. (partial specialization with
 * Empty edge data.)
 *
 * @tparam VID_T
 */
template <typename VID_T>
struct Nbr<VID_T, EmptyType> {
  DEV_HOST Nbr() : neighbor() {}
  DEV_HOST explicit Nbr(const VID_T& nbr_) : neighbor(nbr_) {}
  DEV_HOST explicit Nbr(const Vertex<VID_T>& nbr_) : neighbor(nbr_) {}
  DEV_HOST Nbr(const Nbr& rhs) : neighbor(rhs.neighbor) {}
  DEV_HOST Nbr(const VID_T& nbr_, const EmptyType&) : neighbor(nbr_) {}
  DEV_HOST Nbr(const Vertex<VID_T>& nbr_, const EmptyType&) : neighbor(nbr_) {}
  DEV_HOST ~Nbr() {}

  DEV_HOST_INLINE Nbr& operator=(const Nbr& rhs) {
    neighbor = rhs.neighbor;
    return *this;
  }

  DEV_HOST_INLINE Vertex<VID_T> get_neighbor() const { return neighbor; }
  DEV_HOST_INLINE const EmptyType& get_data() const { return data; }

  DEV_HOST_INLINE VID_T get_neighbor_lid() const { return neighbor.GetValue(); }

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
  DEV_HOST AdjList() {}
  DEV_HOST AdjList(NbrT* b, NbrT* e) : begin_(b), end_(e) {}
  DEV_HOST ~AdjList() {}

  DEV_HOST_INLINE bool Empty() const { return begin_ == end_; }

  DEV_HOST_INLINE bool NotEmpty() const { return !Empty(); }

  DEV_HOST_INLINE size_t Size() const { return end_ - begin_; }

  class iterator {
    using pointer_type = NbrT*;
    using reference_type = NbrT&;

   private:
    NbrT* current_;

   public:
    DEV_HOST iterator() noexcept : current_() {}
    DEV_HOST explicit iterator(const pointer_type& c) noexcept : current_(c) {}
    DEV_HOST reference_type operator*() const noexcept { return *current_; }
    DEV_HOST pointer_type operator->() const noexcept { return current_; }

    DEV_HOST iterator& operator++() noexcept {
      ++current_;
      return *this;
    }

    DEV_HOST iterator operator++(int) noexcept { return iterator(current_++); }

    DEV_HOST iterator& operator--() noexcept {
      --current_;
      return *this;
    }

    DEV_HOST iterator operator--(int) noexcept { return iterator(current_--); }

    DEV_HOST iterator operator+(size_t offset) noexcept {
      return iterator(current_ + offset);
    }

    DEV_HOST bool operator==(const iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    DEV_HOST bool operator!=(const iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  class const_iterator {
    using pointer_type = const NbrT*;
    using reference_type = const NbrT&;

   private:
    const NbrT* current_;

   public:
    DEV_HOST const_iterator() noexcept : current_() {}
    DEV_HOST explicit const_iterator(const pointer_type& c) noexcept
        : current_(c) {}
    DEV_HOST reference_type operator*() const noexcept { return *current_; }
    DEV_HOST pointer_type operator->() const noexcept { return current_; }

    DEV_HOST const_iterator& operator++() noexcept {
      ++current_;
      return *this;
    }

    DEV_HOST const_iterator operator++(int) noexcept {
      return const_iterator(current_++);
    }

    DEV_HOST const_iterator& operator--() noexcept {
      --current_;
      return *this;
    }

    DEV_HOST const_iterator operator--(int) noexcept {
      return const_iterator(current_--);
    }

    DEV_HOST const_iterator operator+(size_t offset) noexcept {
      return const_iterator(current_ + offset);
    }

    DEV_HOST bool operator==(const const_iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    DEV_HOST bool operator!=(const const_iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  DEV_HOST iterator begin() { return iterator(begin_); }
  DEV_HOST iterator end() { return iterator(end_); }

  DEV_HOST const_iterator begin() const { return const_iterator(begin_); }
  DEV_HOST const_iterator end() const { return const_iterator(end_); }

  DEV_HOST NbrT* begin_pointer() { return begin_; }
  DEV_HOST const NbrT* begin_pointer() const { return begin_; }

  DEV_HOST NbrT* end_pointer() { return end_; }
  DEV_HOST const NbrT* end_pointer() const { return end_; }

  DEV_HOST bool empty() const { return end_ == begin_; }

 private:
  NbrT* begin_;
  NbrT* end_;
};

template <typename VID_T, typename EDATA_T, typename PRED_T>
class FilterAdjList {
  using NbrT = Nbr<VID_T, EDATA_T>;

 public:
  FilterAdjList(NbrT* b, NbrT* e, PRED_T pred)
      : begin_(b), end_(e), pred_(pred) {
    while (begin_ != end_ && !pred_(*begin_)) {
      ++begin_;
    }
  }
  ~FilterAdjList() {}

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return !Empty(); }

  class iterator {
    using pointer_type = NbrT*;
    using reference_type = NbrT&;

   private:
    NbrT* current_;
    NbrT* end_;
    const PRED_T& pred_;

   public:
    explicit iterator(const pointer_type& c, const pointer_type& end,
                      const PRED_T& pred) noexcept
        : current_(c), end_(end), pred_(pred) {}
    reference_type operator*() const noexcept { return *current_; }
    pointer_type operator->() const noexcept { return current_; }

    iterator& operator++() noexcept {
      do {
        ++current_;
      } while ((current_ != end_) && !pred_(*current_));
      return *this;
    }

    iterator operator++(int) noexcept {
      NbrT* old_current = current_;
      do {
        ++current_;
      } while ((current_ != end_) && !pred_(*current_));
      return iterator(old_current, end_, pred_);
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
    const NbrT* end_;
    const PRED_T& pred_;

   public:
    explicit const_iterator(const pointer_type& c, const pointer_type& end,
                            const PRED_T& pred) noexcept
        : current_(c), end_(end), pred_(pred) {}
    reference_type operator*() const noexcept { return *current_; }
    pointer_type operator->() const noexcept { return current_; }

    const_iterator& operator++() noexcept {
      do {
        ++current_;
      } while ((current_ != end_) && !pred_(*current_));
      return *this;
    }

    const_iterator operator++(int) noexcept {
      const NbrT* old_current = current_;
      do {
        ++current_;
      } while ((current_ != end_) && !pred_(*current_));
      return const_iterator(old_current, end_, pred_);
    }

    bool operator==(const const_iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    bool operator!=(const const_iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  iterator begin() { return iterator(begin_, end_, pred_); }
  iterator end() { return iterator(end_, end_, pred_); }

  const_iterator begin() const { return const_iterator(begin_, end_, pred_); }
  const_iterator end() const { return const_iterator(end_, end_, pred_); }

  bool empty() const { return end_ == begin_; }

 private:
  NbrT* begin_;
  NbrT* end_;
  PRED_T pred_;
};

template <typename VID_T, typename EDATA_T, typename PRED_T>
class FilterConstAdjList {
  using NbrT = Nbr<VID_T, EDATA_T>;

 public:
  FilterConstAdjList(const NbrT* b, const NbrT* e, PRED_T pred)
      : begin_(b), end_(e), pred_(pred) {
    while (begin_ != end_ && !pred_(*begin_)) {
      ++begin_;
    }
  }
  ~FilterConstAdjList() {}

  inline bool Empty() const { return begin_ == end_; }

  inline bool NotEmpty() const { return !Empty(); }

  class const_iterator {
    using pointer_type = const NbrT*;
    using reference_type = const NbrT&;

   private:
    const NbrT* current_;
    const NbrT* end_;
    const PRED_T& pred_;

   public:
    explicit const_iterator(const pointer_type& c, const pointer_type& end,
                            const PRED_T& pred) noexcept
        : current_(c), end_(end), pred_(pred) {}
    reference_type operator*() const noexcept { return *current_; }
    pointer_type operator->() const noexcept { return current_; }

    const_iterator& operator++() noexcept {
      do {
        ++current_;
      } while ((current_ != end_) && !pred_(*current_));
      return *this;
    }

    const_iterator operator++(int) noexcept {
      const NbrT* old_current = current_;
      do {
        ++current_;
      } while ((current_ != end_) && !pred_(*current_));
      return const_iterator(old_current, end_, pred_);
    }

    bool operator==(const const_iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    bool operator!=(const const_iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  const_iterator begin() const { return const_iterator(begin_, end_, pred_); }
  const_iterator end() const { return const_iterator(end_, end_, pred_); }

  bool empty() const { return end_ == begin_; }

 private:
  const NbrT* begin_;
  const NbrT* end_;
  PRED_T pred_;
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
  DEV_HOST ConstAdjList() {}
  DEV_HOST ConstAdjList(const NbrT* b, const NbrT* e) : begin_(b), end_(e) {}
  DEV_HOST ~ConstAdjList() {}

  DEV_HOST_INLINE bool Empty() const { return begin_ == end_; }

  DEV_HOST_INLINE bool NotEmpty() const { return !Empty(); }

  DEV_HOST_INLINE size_t Size() const { return end_ - begin_; }

  class const_iterator {
    using pointer_type = const NbrT*;
    using reference_type = const NbrT&;

   private:
    const NbrT* current_;

   public:
    const_iterator() noexcept : current_() {}
    DEV_HOST explicit const_iterator(const pointer_type& c) noexcept
        : current_(c) {}
    DEV_HOST reference_type operator*() const noexcept { return *current_; }
    DEV_HOST pointer_type operator->() const noexcept { return current_; }

    DEV_HOST const_iterator& operator++() noexcept {
      ++current_;
      return *this;
    }

    DEV_HOST const_iterator operator++(int) noexcept {
      return const_iterator(current_++);
    }

    DEV_HOST const_iterator& operator--() noexcept {
      --current_;
      return *this;
    }

    DEV_HOST const_iterator operator--(int) noexcept {
      return const_iterator(current_--);
    }

    DEV_HOST const_iterator operator+(size_t offset) noexcept {
      return const_iterator(current_ + offset);
    }

    DEV_HOST bool operator==(const const_iterator& rhs) noexcept {
      return current_ == rhs.current_;
    }
    DEV_HOST bool operator!=(const const_iterator& rhs) noexcept {
      return current_ != rhs.current_;
    }
  };

  DEV_HOST const_iterator begin() const { return const_iterator(begin_); }
  DEV_HOST const_iterator end() const { return const_iterator(end_); }

  DEV_HOST const NbrT* begin_pointer() const { return begin_; }

  DEV_HOST const NbrT* end_pointer() const { return end_; }

  DEV_HOST bool empty() const { return end_ == begin_; }

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
  DEV_HOST DestList(const fid_t* begin_, const fid_t* end_)
      : begin(begin_), end(end_) {}
  DEV_HOST_INLINE bool Empty() { return begin == end; }
  DEV_HOST_INLINE bool NotEmpty() { return !Empty(); }
  const fid_t* begin;
  const fid_t* end;
};

}  // namespace grape

#endif  // GRAPE_GRAPH_ADJ_LIST_H_
