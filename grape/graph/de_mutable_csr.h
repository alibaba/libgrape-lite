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

#ifndef GRAPE_GRAPH_DE_MUTABLE_CSR_H_
#define GRAPE_GRAPH_DE_MUTABLE_CSR_H_

#include <algorithm>

#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/mutable_csr.h"

namespace grape {

template <typename VID_T, typename NBR_T>
class DeMutableCSR;

template <typename VID_T, typename NBR_T>
class DeMutableCSRBuilder {};

template <typename VID_T, typename EDATA_T>
class DeMutableCSRBuilder<VID_T, Nbr<VID_T, EDATA_T>> {
  using vid_t = VID_T;

 public:
  using vertex_range_t = DualVertexRange<VID_T>;

  DeMutableCSRBuilder() {}
  ~DeMutableCSRBuilder() {}

  void init(VID_T min_id, VID_T max_head, VID_T min_tail, VID_T max_id,
            bool dedup = false) {
    min_id_ = min_id;
    max_id_ = max_id;
    max_head_id_ = max_head;
    min_tail_id_ = min_tail;
    dedup_ = dedup;

    head_builder_.init(max_head_id_ - min_id_);
    tail_builder_.init(max_id_ - min_tail_id_);
  }

  void init(const vertex_range_t& range, bool dedup = false) {
    min_id_ = range.head().begin_value();
    max_id_ = range.tail().end_value();
    max_head_id_ = range.head().end_value();
    min_tail_id_ = range.tail().begin_value();
    dedup_ = dedup;

    head_builder_.init(max_head_id_ - min_id_);
    tail_builder_.init(max_id_ - min_tail_id_);
  }

  void inc_degree(VID_T i) {
    if (in_head(i)) {
      head_builder_.inc_degree(head_index(i));
    } else {
      tail_builder_.inc_degree(tail_index(i));
    }
  }

  void build_offsets() {
    head_builder_.build_offsets();
    tail_builder_.build_offsets();
  }

  void add_edge(VID_T src, const Nbr<VID_T, EDATA_T>& nbr) {
    if (in_head(src)) {
      head_builder_.add_edge(head_index(src), nbr);
    } else {
      tail_builder_.add_edge(tail_index(src), nbr);
    }
  }

  void finish(DeMutableCSR<VID_T, Nbr<VID_T, EDATA_T>>& ret) {
    ret.min_id_ = min_id_;
    ret.max_id_ = max_id_;
    ret.max_head_id_ = max_head_id_;
    ret.min_tail_id_ = min_tail_id_;
    ret.dedup_ = dedup_;

    head_builder_.finish(ret.head_);
    tail_builder_.finish(ret.tail_);

    if (dedup_) {
      VID_T head_num = ret.head_.vertex_num();
      for (VID_T i = 0; i < head_num; ++i) {
        ret.head_.dedup_neighbors(i);
      }
      VID_T tail_num = ret.tail_.vertex_num();
      for (VID_T i = 0; i < tail_num; ++i) {
        ret.tail_.dedup_neighbors(i);
      }
    } else {
      VID_T head_num = ret.head_.vertex_num();
      for (VID_T i = 0; i < head_num; ++i) {
        ret.head_.sort_neighbors(i);
      }
      VID_T tail_num = ret.tail_.vertex_num();
      for (VID_T i = 0; i < tail_num; ++i) {
        ret.tail_.sort_neighbors(i);
      }
    }
  }

 private:
  inline bool in_head(vid_t i) const { return i < max_head_id_; }

  inline vid_t head_index(vid_t i) const { return i - min_id_; }

  inline vid_t tail_index(vid_t i) const { return max_id_ - i - 1; }

  VID_T min_id_;
  VID_T max_id_;

  VID_T max_head_id_;
  VID_T min_tail_id_;
  bool dedup_;

  MutableCSRBuilder<VID_T, Nbr<VID_T, EDATA_T>> head_builder_, tail_builder_;
};

template <typename VID_T, typename NBR_T>
class DeMutableCSR {};

template <typename VID_T, typename EDATA_T>
class DeMutableCSR<VID_T, Nbr<VID_T, EDATA_T>> {
 public:
  using vid_t = VID_T;
  using nbr_t = Nbr<vid_t, EDATA_T>;
  using edge_t = Edge<vid_t, EDATA_T>;
  using adj_list_t = AdjList<vid_t, EDATA_T>;

  static constexpr double dense_threshold = 0.003;

  DeMutableCSR() {}

  DeMutableCSR(vid_t from, vid_t to, bool dedup = false)
      : min_id_(from),
        max_id_(to),
        max_head_id_(from),
        min_tail_id_(to),
        dedup_(dedup) {}

  VID_T vertex_num() const {
    return (max_id_ - min_tail_id_) + (max_head_id_ - min_id_);
  }

  bool empty() const { return head_.empty() && tail_.empty(); }

  size_t edge_num() const { return head_.edge_num() + tail_.edge_num(); }

  size_t head_edge_num() const { return head_.edge_num(); }

  size_t tail_edge_num() const { return tail_.edge_num(); }

  int degree(VID_T i) const {
    return in_head(i) ? head_.degree(head_index(i))
                      : tail_.degree(tail_index(i));
  }

  void remove_vertex(VID_T i) {
    if (in_head(i)) {
      head_.remove_vertex(head_index(i));
    } else {
      tail_.remove_vertex(tail_index(i));
    }
  }

  bool is_empty(VID_T i) const {
    return in_head(i) ? head_.is_empty(head_index(i))
                      : tail_.is_empty(tail_index(i));
  }

  inline bool in_head(vid_t i) const { return i < max_head_id_; }

  inline vid_t head_index(vid_t i) const { return i - min_id_; }

  inline vid_t tail_index(vid_t i) const { return max_id_ - i - 1; }

  inline vid_t get_index(vid_t i) const {
    return in_head(i) ? head_index(i) : tail_index(i);
  }

  nbr_t* get_begin(vid_t i) {
    return in_head(i) ? head_.get_begin(head_index(i))
                      : tail_.get_begin(tail_index(i));
  }

  const nbr_t* get_begin(vid_t i) const {
    return in_head(i) ? head_.get_begin(head_index(i))
                      : tail_.get_begin(tail_index(i));
  }

  nbr_t* get_end(vid_t i) {
    return in_head(i) ? head_.get_end(head_index(i))
                      : tail_.get_end(tail_index(i));
  }

  const nbr_t* get_end(vid_t i) const {
    return in_head(i) ? head_.get_end(head_index(i))
                      : tail_.get_end(tail_index(i));
  }

  nbr_t* find(VID_T i, VID_T nbr) {
    return in_head(i) ? head_.find(head_index(i), nbr)
                      : tail_.find(tail_index(i), nbr);
  }

  const nbr_t* find(VID_T i, VID_T nbr) const {
    return in_head(i) ? head_.find(head_index(i), nbr)
                      : tail_.find(tail_index(i), nbr);
  }

  nbr_t* binary_find(VID_T i, VID_T nbr) {
    return in_head(i) ? head_.binary_find(head_index(i), nbr)
                      : tail_.binary_find(tail_index(i), nbr);
  }
  const nbr_t* binary_find(VID_T i, VID_T nbr) const {
    return in_head(i) ? head_.binary_find(head_index(i), nbr)
                      : tail_.binary_find(tail_index(i), nbr);
  }

  void add_vertices(vid_t to_head, vid_t to_tail) {
    if (to_head != 0) {
      max_head_id_ += to_head;
      vid_t head_num = max_head_id_ - min_id_;
      head_.reserve_vertices(head_num);
    }

    if (to_tail != 0) {
      min_tail_id_ -= to_tail;
      vid_t tail_num = max_id_ - min_tail_id_;
      tail_.reserve_vertices(tail_num);
    }
  }

  void add_edges(const std::vector<edge_t>& edges) {
    double rate =
        static_cast<double>(edges.size()) / static_cast<double>(edge_num());
    if (rate < dense_threshold) {
      add_edges_sparse(edges);
    } else {
      add_edges_dense(edges);
    }
  }

  void add_forward_edges(const std::vector<edge_t>& edges) {
    double rate =
        static_cast<double>(edges.size()) / static_cast<double>(edge_num());
    if (rate < dense_threshold) {
      add_forward_edges_sparse(edges);
    } else {
      add_forward_edges_dense(edges);
    }
  }

  void add_reversed_edges(const std::vector<edge_t>& edges) {
    double rate =
        static_cast<double>(edges.size()) / static_cast<double>(edge_num());
    if (rate < dense_threshold) {
      add_reversed_edges_sparse(edges);
    } else {
      add_reversed_edges_dense(edges);
    }
  }

  void init_head_and_tail(vid_t min, vid_t max, bool dedup = false) {
    min_id_ = max_head_id_ = min;
    max_id_ = min_tail_id_ = max;
    dedup_ = dedup;
  }

  // break the operation of `add_edges` into 3 steps:
  // (1) `reserve_edges` for reserving space (capacity) of edges
  // (2) `put_edge` for inserting an edge, which can be parallel
  // (3) `sort_neighbors` for cleanning up the nerghbors

  // `degree_to_add` is indexed by the real index,
  // and the caller is responsible for conversion
  void reserve_edges_dense(const std::vector<int>& head_degree_to_add,
                           const std::vector<int>& tail_degree_to_add) {
    head_.reserve_edges_dense(head_degree_to_add);
    tail_.reserve_edges_dense(tail_degree_to_add);
  }

  void reserve_edges_sparse(const std::map<vid_t, int>& degree_to_add) {
    std::map<vid_t, int> head_degree_to_add, tail_degree_to_add;

    for (const auto& pair : degree_to_add) {
      if (in_head(pair.first)) {
        head_degree_to_add.insert(
            std::make_pair(head_index(pair.first), pair.second));
      } else {
        tail_degree_to_add.insert(
            std::make_pair(tail_index(pair.first), pair.second));
      }
    }
    head_.reserve_edges_sparse(head_degree_to_add);
    tail_.reserve_edges_sparse(tail_degree_to_add);
  }

  nbr_t* put_edge(vid_t src, const nbr_t& value) {
    if (in_head(src)) {
      return head_.put_edge(head_index(src), value);
    } else {
      return tail_.put_edge(tail_index(src), value);
    }
  }

  nbr_t* put_edge(vid_t src, nbr_t&& value) {
    if (in_head(src)) {
      return head_.put_edge(head_index(src), value);
    } else {
      return tail_.put_edge(tail_index(src), value);
    }
  }

  // `degree_to_add` is indexed by the real index,
  // and the caller is responsible for conversion
  void sort_neighbors_dense(const std::vector<int>& head_degree_to_add,
                            const std::vector<int>& tail_degree_to_add) {
    head_.sort_neighbors_dense(head_degree_to_add);
    tail_.sort_neighbors_dense(tail_degree_to_add);
  }

  void sort_neighbors_sparse(const std::map<vid_t, int>& degree_to_add) {
    std::map<vid_t, int> head_degree_to_add, tail_degree_to_add;

    for (const auto& pair : degree_to_add) {
      if (in_head(pair.first)) {
        head_degree_to_add.insert(
            std::make_pair(head_index(pair.first), pair.second));
      } else {
        tail_degree_to_add.insert(
            std::make_pair(tail_index(pair.first), pair.second));
      }
    }
    head_.sort_neighbors_sparse(head_degree_to_add);
    tail_.sort_neighbors_sparse(tail_degree_to_add);
  }

  void remove_edges(const std::vector<edge_t>& edges) {
    vid_t head_num = max_head_id_ - min_id_;
    vid_t tail_num = max_id_ - min_tail_id_;
    std::vector<bool> head_modified(head_num, false),
        tail_modified(tail_num, false);
    if (dedup_) {
      for (auto& e : edges) {
        vid_t src = e.src;
        if (in_head(src)) {
          vid_t index = head_index(src);
          head_modified[index] =
              head_.remove_one_with_tomb(index, e.dst) || head_modified[index];
        } else {
          vid_t index = tail_index(src);
          tail_modified[index] =
              tail_.remove_one_with_tomb(index, e.dst) || tail_modified[index];
        }
      }
    } else {
      for (auto& e : edges) {
        vid_t src = e.src;
        if (in_head(src)) {
          vid_t index = head_index(src);
          head_modified[index] =
              head_.remove_with_tomb(index, e.dst) || head_modified[index];
        } else {
          vid_t index = tail_index(src);
          tail_modified[index] =
              tail_.remove_with_tomb(index, e.dst) || tail_modified[index];
        }
      }
    }
    for (vid_t i = 0; i < head_num; ++i) {
      if (head_modified[i]) {
        head_.remove_tombs(i);
      }
    }
    for (vid_t i = 0; i < tail_num; ++i) {
      if (tail_modified[i]) {
        tail_.remove_tombs(i);
      }
    }
  }

  void remove_edges(const std::vector<std::pair<vid_t, vid_t>>& edges) {
    vid_t head_num = max_head_id_ - min_id_;
    vid_t tail_num = max_id_ - min_tail_id_;
    std::vector<bool> head_modified(head_num, false),
        tail_modified(tail_num, false);
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    if (dedup_) {
      for (auto& e : edges) {
        if (e.first == sentinel) {
          continue;
        }
        vid_t src = e.first;
        if (in_head(src)) {
          vid_t index = head_index(src);
          head_modified[index] = head_.remove_one_with_tomb(index, e.second) ||
                                 head_modified[index];
        } else {
          vid_t index = tail_index(src);
          tail_modified[index] = tail_.remove_one_with_tomb(index, e.second) ||
                                 tail_modified[index];
        }
      }
    } else {
      for (auto& e : edges) {
        if (e.first == sentinel) {
          continue;
        }
        vid_t src = e.first;
        if (in_head(src)) {
          vid_t index = head_index(src);
          head_modified[index] =
              head_.remove_with_tomb(index, e.second) || head_modified[index];
        } else {
          vid_t index = tail_index(src);
          tail_modified[index] =
              tail_.remove_with_tomb(index, e.second) || tail_modified[index];
        }
      }
    }
    for (vid_t i = 0; i < head_num; ++i) {
      if (head_modified[i]) {
        head_.remove_tombs(i);
      }
    }
    for (vid_t i = 0; i < tail_num; ++i) {
      if (tail_modified[i]) {
        tail_.remove_tombs(i);
      }
    }
  }

  void remove_reversed_edges(
      const std::vector<std::pair<vid_t, vid_t>>& edges) {
    vid_t head_num = max_head_id_ - min_id_;
    vid_t tail_num = max_id_ - min_tail_id_;
    std::vector<bool> head_modified(head_num, false),
        tail_modified(tail_num, false);
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    if (dedup_) {
      for (auto& e : edges) {
        if (e.first == sentinel) {
          continue;
        }
        vid_t src = e.second;
        if (in_head(src)) {
          vid_t index = head_index(src);
          head_modified[index] = head_.remove_one_with_tomb(index, e.first) ||
                                 head_modified[index];
        } else {
          vid_t index = tail_index(src);
          tail_modified[index] = tail_.remove_one_with_tomb(index, e.first) ||
                                 tail_modified[index];
        }
      }
    } else {
      for (auto& e : edges) {
        if (e.first == sentinel) {
          continue;
        }
        vid_t src = e.second;
        if (in_head(src)) {
          vid_t index = head_index(src);
          head_modified[index] =
              head_.remove_with_tomb(index, e.first) || head_modified[index];
        } else {
          vid_t index = tail_index(src);
          tail_modified[index] =
              tail_.remove_with_tomb(index, e.first) || tail_modified[index];
        }
      }
    }
    for (vid_t i = 0; i < head_num; ++i) {
      if (head_modified[i]) {
        head_.remove_tombs(i);
      }
    }
    for (vid_t i = 0; i < tail_num; ++i) {
      if (tail_modified[i]) {
        tail_.remove_tombs(i);
      }
    }
  }

  template <typename FUNC_T>
  void remove_if(const FUNC_T& func) {
    head_.remove_if(func);
    tail_.remove_if(func);
  }

  void update_edges(const std::vector<edge_t>& edges) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    if (dedup_) {
      for (auto& e : edges) {
        if (e.src == sentinel) {
          continue;
        }
        vid_t src = e.src;
        if (in_head(src)) {
          head_.update_one(head_index(src), e.dst, e.edata);
        } else {
          tail_.update_one(tail_index(src), e.dst, e.edata);
        }
      }
    } else {
      for (auto& e : edges) {
        if (e.src == sentinel) {
          continue;
        }
        vid_t src = e.src;
        if (in_head(src)) {
          head_.update(head_index(src), e.dst, e.edata);
        } else {
          tail_.update(tail_index(src), e.dst, e.edata);
        }
      }
    }
  }

  void update_reversed_edges(const std::vector<edge_t>& edges) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    if (dedup_) {
      for (auto& e : edges) {
        if (e.src == sentinel) {
          continue;
        }
        vid_t src = e.dst;
        if (in_head(src)) {
          head_.update_one(head_index(src), e.src, e.edata);
        } else {
          tail_.update_one(tail_index(src), e.src, e.edata);
        }
      }
    } else {
      for (auto& e : edges) {
        if (e.src == sentinel) {
          continue;
        }
        vid_t src = e.dst;
        if (in_head(src)) {
          head_.update(head_index(src), e.src, e.edata);
        } else {
          tail_.update(tail_index(src), e.src, e.edata);
        }
      }
    }
  }

  void clear_edges() {
    head_.clear_edges();
    tail_.clear_edges();
  }

  template <typename IOADAPTOR_T>
  void Serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    InArchive ia;
    ia << min_id_ << max_id_ << max_head_id_ << min_tail_id_ << dedup_;
    CHECK(writer->WriteArchive(ia));
    head_.template Serialize<IOADAPTOR_T>(writer);
    tail_.template Serialize<IOADAPTOR_T>(writer);
  }

  template <typename IOADAPTOR_T>
  void Deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    OutArchive oa;
    CHECK(reader->ReadArchive(oa));
    oa >> min_id_ >> max_id_ >> max_head_id_ >> min_tail_id_ >> dedup_;
    head_.template Deserialize<IOADAPTOR_T>(reader);
    tail_.template Deserialize<IOADAPTOR_T>(reader);
  }

 private:
  void add_reversed_edges_dense(const std::vector<edge_t>& edges) {
    vid_t head_num = max_head_id_ - min_id_;
    vid_t tail_num = max_id_ - min_tail_id_;

    std::vector<int> head_degree_to_add(head_num, 0),
        tail_degree_to_add(tail_num, 0);
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.dst)) {
        ++head_degree_to_add[head_index(e.dst)];
      } else {
        ++tail_degree_to_add[tail_index(e.dst)];
      }
    }

    head_.reserve_edges_dense(head_degree_to_add);
    tail_.reserve_edges_dense(tail_degree_to_add);

    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.dst)) {
        head_.put_edge(head_index(e.dst), nbr_t(e.src, e.edata));
      } else {
        tail_.put_edge(tail_index(e.dst), nbr_t(e.src, e.edata));
      }
    }

    if (dedup_) {
      head_.dedup_neighbors_dense(head_degree_to_add);
      tail_.dedup_neighbors_dense(tail_degree_to_add);
    } else {
      head_.sort_neighbors_dense(head_degree_to_add);
      tail_.sort_neighbors_dense(tail_degree_to_add);
    }
  }

  void add_forward_edges_dense(const std::vector<edge_t>& edges) {
    vid_t head_num = max_head_id_ - min_id_;
    vid_t tail_num = max_id_ - min_tail_id_;

    std::vector<int> head_degree_to_add(head_num, 0),
        tail_degree_to_add(tail_num, 0);
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        ++head_degree_to_add[head_index(e.src)];
      } else {
        ++tail_degree_to_add[tail_index(e.src)];
      }
    }

    head_.reserve_edges_dense(head_degree_to_add);
    tail_.reserve_edges_dense(tail_degree_to_add);

    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        head_.put_edge(head_index(e.src), nbr_t(e.dst, e.edata));
      } else {
        tail_.put_edge(tail_index(e.src), nbr_t(e.dst, e.edata));
      }
    }

    if (dedup_) {
      head_.dedup_neighbors_dense(head_degree_to_add);
      tail_.dedup_neighbors_dense(tail_degree_to_add);
    } else {
      head_.sort_neighbors_dense(head_degree_to_add);
      tail_.sort_neighbors_dense(tail_degree_to_add);
    }
  }

  void add_edges_dense(const std::vector<edge_t>& edges) {
    vid_t head_num = max_head_id_ - min_id_;
    vid_t tail_num = max_id_ - min_tail_id_;

    std::vector<int> head_degree_to_add(head_num, 0),
        tail_degree_to_add(tail_num, 0);
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        ++head_degree_to_add[head_index(e.src)];
      } else {
        ++tail_degree_to_add[tail_index(e.src)];
      }
      if (in_head(e.dst)) {
        ++head_degree_to_add[head_index(e.dst)];
      } else {
        ++tail_degree_to_add[tail_index(e.dst)];
      }
    }

    head_.reserve_edges_dense(head_degree_to_add);
    tail_.reserve_edges_dense(tail_degree_to_add);

    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        head_.put_edge(head_index(e.src), nbr_t(e.dst, e.edata));
      } else {
        tail_.put_edge(tail_index(e.src), nbr_t(e.dst, e.edata));
      }
      if (in_head(e.dst)) {
        head_.put_edge(head_index(e.dst), nbr_t(e.src, e.edata));
      } else {
        tail_.put_edge(tail_index(e.dst), nbr_t(e.src, e.edata));
      }
    }

    if (dedup_) {
      head_.dedup_neighbors_dense(head_degree_to_add);
      tail_.dedup_neighbors_dense(tail_degree_to_add);
    } else {
      head_.sort_neighbors_dense(head_degree_to_add);
      tail_.sort_neighbors_dense(tail_degree_to_add);
    }
  }

  void add_edges_sparse(const std::vector<edge_t>& edges) {
    std::map<vid_t, int> head_degree_to_add, tail_degree_to_add;
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        ++head_degree_to_add[head_index(e.src)];
      } else {
        ++tail_degree_to_add[tail_index(e.src)];
      }
      if (in_head(e.dst)) {
        ++head_degree_to_add[head_index(e.dst)];
      } else {
        ++tail_degree_to_add[tail_index(e.dst)];
      }
    }

    head_.reserve_edges_sparse(head_degree_to_add);
    tail_.reserve_edges_sparse(tail_degree_to_add);

    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        head_.put_edge(head_index(e.src), nbr_t(e.dst, e.edata));
      } else {
        tail_.put_edge(tail_index(e.src), nbr_t(e.dst, e.edata));
      }
      if (in_head(e.dst)) {
        head_.put_edge(head_index(e.dst), nbr_t(e.src, e.edata));
      } else {
        tail_.put_edge(tail_index(e.dst), nbr_t(e.src, e.edata));
      }
    }

    if (dedup_) {
      head_.dedup_neighbors_sparse(head_degree_to_add);
      tail_.dedup_neighbors_sparse(tail_degree_to_add);
    } else {
      head_.sort_neighbors_sparse(head_degree_to_add);
      tail_.sort_neighbors_sparse(tail_degree_to_add);
    }
  }

  void add_forward_edges_sparse(const std::vector<edge_t>& edges) {
    std::map<vid_t, int> head_degree_to_add, tail_degree_to_add;
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        ++head_degree_to_add[head_index(e.src)];
      } else {
        ++tail_degree_to_add[tail_index(e.src)];
      }
    }

    head_.reserve_edges_sparse(head_degree_to_add);
    tail_.reserve_edges_sparse(tail_degree_to_add);

    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.src)) {
        head_.put_edge(head_index(e.src), nbr_t(e.dst, e.edata));
      } else {
        tail_.put_edge(tail_index(e.src), nbr_t(e.dst, e.edata));
      }
    }

    if (dedup_) {
      head_.dedup_neighbors_sparse(head_degree_to_add);
      tail_.dedup_neighbors_sparse(tail_degree_to_add);
    } else {
      head_.sort_neighbors_sparse(head_degree_to_add);
      tail_.sort_neighbors_sparse(tail_degree_to_add);
    }
  }

  void add_reversed_edges_sparse(const std::vector<edge_t>& edges) {
    std::map<vid_t, int> head_degree_to_add, tail_degree_to_add;
    static constexpr VID_T invalid_vid = std::numeric_limits<VID_T>::max();
    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.dst)) {
        ++head_degree_to_add[head_index(e.dst)];
      } else {
        ++tail_degree_to_add[tail_index(e.dst)];
      }
    }

    head_.reserve_edges_sparse(head_degree_to_add);
    tail_.reserve_edges_sparse(tail_degree_to_add);

    for (auto& e : edges) {
      if (e.src == invalid_vid) {
        continue;
      }
      if (in_head(e.dst)) {
        head_.put_edge(head_index(e.dst), nbr_t(e.src, e.edata));
      } else {
        tail_.put_edge(tail_index(e.dst), nbr_t(e.src, e.edata));
      }
    }

    if (dedup_) {
      head_.dedup_neighbors_sparse(head_degree_to_add);
      tail_.dedup_neighbors_sparse(tail_degree_to_add);
    } else {
      head_.sort_neighbors_sparse(head_degree_to_add);
      tail_.sort_neighbors_sparse(tail_degree_to_add);
    }
  }

  template <typename _VID_T, typename _NBR_T>
  friend class DeMutableCSRBuilder;

  vid_t min_id_;
  vid_t max_id_;

  vid_t max_head_id_;
  vid_t min_tail_id_;
  bool dedup_;

  MutableCSR<VID_T, Nbr<VID_T, EDATA_T>> head_, tail_;
};

}  // namespace grape

#endif  // GRAPE_GRAPH_DE_MUTABLE_CSR_H_
