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

#ifndef GRAPE_GRAPH_MUTABLE_CSR_H_
#define GRAPE_GRAPH_MUTABLE_CSR_H_

#include <assert.h>

#include <algorithm>
#include <map>
#include <vector>

#include "grape/graph/adj_list.h"
#include "grape/graph/mutable_csr_impl.h"

namespace grape {

template <typename VID_T, typename NBR_T>
class MutableCSR;

template <typename VID_T, typename NBR_T>
class MutableCSRBuilder {};

template <typename VID_T, typename EDATA_T>
class MutableCSRBuilder<VID_T, Nbr<VID_T, EDATA_T>> {
  using vid_t = VID_T;
  using nbr_t = Nbr<vid_t, EDATA_T>;
  using adj_list_t = mutable_csr_impl::AdjList<nbr_t>;
  static constexpr double relax_rate = 1.5;

 public:
  using vertex_range_t = VertexRange<VID_T>;

  MutableCSRBuilder() {}
  ~MutableCSRBuilder() {}

  void init(VID_T vnum) {
    vnum_ = vnum;
    degree_.clear();
    degree_.resize(vnum, 0);
  }

  void init(const vertex_range_t& range) {
    assert(range.begin_value() == 0);
    vnum_ = range.size();
    degree_.clear();
    degree_.resize(vnum_, 0);
  }

  void inc_degree(VID_T i) {
    if (i < vnum_) {
      ++degree_[i];
    }
  }
  void resize(VID_T vnum) {
    vnum_ = vnum;
    degree_.resize(vnum_, 0);
  }

  VID_T vertex_num() const { return vnum_; }

  void build_offsets() {
    size_t total_capacity = 0;
    for (auto d : degree_) {
      total_capacity += d * relax_rate;
    }
    buffer_.resize(total_capacity);
    nbr_t* ptr = buffer_.data();
    adj_lists_.resize(vnum_);
    capacity_.resize(vnum_);
    iter_.resize(vnum_);
    for (vid_t i = 0; i < vnum_; ++i) {
      iter_[i] = adj_lists_[i].begin = ptr;
      adj_lists_[i].end = ptr + degree_[i];
      capacity_[i] = degree_[i] * relax_rate;
      ptr += capacity_[i];
    }
  }

  void add_edge(VID_T src, const nbr_t& nbr) {
    if (src < vnum_) {
      nbr_t* ptr = iter_[src]++;
      *ptr = nbr;
    }
  }

  template <typename FUNC_T>
  void sort(const FUNC_T& func) {
    for (VID_T i = 0; i < vnum_; ++i) {
      std::sort(adj_lists_[i].begin, adj_lists_[i].end, func);
    }
  }

  void finish(MutableCSR<VID_T, Nbr<VID_T, EDATA_T>>& ret) {
    if (vnum_ == 0) {
      ret.capacity_.clear();
      ret.prev_.clear();
      ret.next_.clear();
      ret.adj_lists_.clear();
      ret.buffers_.clear();
      return;
    }
    ret.capacity_.swap(capacity_);
    ret.prev_.resize(vnum_);
    ret.next_.resize(vnum_);
    for (vid_t i = 0; i < vnum_; ++i) {
      ret.prev_[i] = i - 1;
      ret.next_[i] = i + 1;
    }
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    ret.prev_[0] = sentinel;
    ret.next_[vnum_ - 1] = sentinel;
    ret.adj_lists_.swap(adj_lists_);
    ret.buffers_.emplace_back(std::move(buffer_));
  }

 private:
  VID_T vnum_;

  std::vector<int> capacity_;
  std::vector<adj_list_t> adj_lists_;
  std::vector<nbr_t*> iter_;
  std::vector<int> degree_;

  mutable_csr_impl::Blob<vid_t, nbr_t> buffer_;
};

template <typename VID_T, typename NBR_T>
class MutableCSR {};

template <typename VID_T, typename EDATA_T>
class MutableCSR<VID_T, Nbr<VID_T, EDATA_T>> {
  using vid_t = VID_T;
  using nbr_t = Nbr<VID_T, EDATA_T>;
  using adj_list_t = mutable_csr_impl::AdjList<nbr_t>;
  static constexpr double relax_rate = 1.5;

 public:
  MutableCSR() {}
  ~MutableCSR() {}

  VID_T vertex_num() const { return adj_lists_.size(); }

  bool empty() const { return adj_lists_.empty(); }

  size_t edge_num() const {
    size_t ret = 0;
    vid_t vnum = vertex_num();
    for (vid_t i = 0; i < vnum; ++i) {
      ret += adj_lists_[i].degree();
    }
    return ret;
  }

  int degree(VID_T i) const { return adj_lists_[i].degree(); }

  bool is_empty(VID_T i) const { return adj_lists_[i].empty(); }

  nbr_t* get_begin(VID_T i) { return adj_lists_[i].begin; }
  const nbr_t* get_begin(VID_T i) const { return adj_lists_[i].begin; }

  nbr_t* get_end(VID_T i) { return adj_lists_[i].end; }
  const nbr_t* get_end(VID_T i) const { return adj_lists_[i].end; }

  nbr_t* find(VID_T i, VID_T nbr) {
    return std::find_if(
        adj_lists_[i].begin, adj_lists_[i].end,
        [&nbr](const nbr_t& item) { return item.neighbor.GetValue() == nbr; });
  }
  const nbr_t* find(VID_T i, VID_T nbr) const {
    return std::find_if(
        adj_lists_[i].begin, adj_lists_[i].end,
        [&nbr](const nbr_t& item) { return item.neighbor.GetValue() == nbr; });
  }

  nbr_t* binary_find(VID_T i, VID_T nbr) {
    return mutable_csr_impl::binary_search_one(adj_lists_[i].begin,
                                               adj_lists_[i].end, nbr);
  }
  const nbr_t* binary_find(VID_T i, VID_T nbr) const {
    return mutable_csr_impl::binary_search_one(adj_lists_[i].begin,
                                               adj_lists_[i].end, nbr);
  }

  void reserve_vertices(vid_t vnum) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    assert(vnum >= vertex_num());
    if (vnum == vertex_num()) {
      return;
    }
    capacity_.resize(vnum, 0);
    prev_.resize(vnum, sentinel);
    next_.resize(vnum, sentinel);
    adj_lists_.resize(vnum);
  }

  void reserve_edges_dense(const std::vector<int>& degree_to_add) {
    vid_t vnum = vertex_num();
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    assert(vnum == degree_to_add.size());
    size_t new_buf_size = 0;
    for (vid_t i = 0; i < vnum; ++i) {
      if (degree_to_add[i] == 0) {
        continue;
      }
      int requirement = adj_lists_[i].degree() + degree_to_add[i];
      if (requirement > capacity_[i]) {
        remove_node(i);
        requirement = requirement * relax_rate;
        new_buf_size += requirement;
        capacity_[i] = -requirement;
      }
    }
    if (new_buf_size != 0) {
      mutable_csr_impl::Blob<vid_t, nbr_t> new_buf(new_buf_size);
      vid_t prev = sentinel;
      nbr_t* begin = new_buf.data();
      for (vid_t i = 0; i < vnum; ++i) {
        if (capacity_[i] < 0) {
          capacity_[i] = -capacity_[i];
          prev_[i] = prev;
          if (prev != sentinel) {
            next_[prev] = i;
          }
          prev = i;
          int old_degree = adj_lists_[i].degree();
          if (old_degree > 0) {
            std::move(adj_lists_[i].begin, adj_lists_[i].end, begin);
          }
          adj_lists_[i].begin = begin;
          adj_lists_[i].end = begin + old_degree;
          begin += capacity_[i];
        }
      }
      if (prev != sentinel) {
        next_[prev] = sentinel;
      }
      buffers_.emplace_back(std::move(new_buf));
    }
  }

  void reserve_edges_sparse(const std::map<vid_t, int>& degree_to_add) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    size_t new_buf_size = 0;
    for (auto& pair : degree_to_add) {
      vid_t cur = pair.first;
      int degree = pair.second;
      int requirement = adj_lists_[cur].degree() + degree;
      if (requirement > capacity_[cur]) {
        remove_node(cur);
        requirement = requirement * relax_rate;
        new_buf_size += requirement;
        capacity_[cur] = -requirement;
      }
    }
    if (new_buf_size != 0) {
      mutable_csr_impl::Blob<vid_t, nbr_t> new_buf(new_buf_size);
      vid_t prev = sentinel;
      nbr_t* begin = new_buf.data();
      for (auto& pair : degree_to_add) {
        vid_t cur = pair.first;
        if (capacity_[cur] < 0) {
          capacity_[cur] = -capacity_[cur];
          prev_[cur] = prev;
          if (prev != sentinel) {
            next_[prev] = cur;
          }
          prev = cur;
          int old_degree = adj_lists_[cur].degree();
          if (old_degree > 0) {
            std::move(adj_lists_[cur].begin, adj_lists_[cur].end, begin);
          }
          adj_lists_[cur].begin = begin;
          adj_lists_[cur].end = begin + old_degree;
          begin += capacity_[cur];
        }
      }
      if (prev != sentinel) {
        next_[prev] = sentinel;
      }
      buffers_.emplace_back(std::move(new_buf));
    }
  }

  nbr_t* put_edge(vid_t src, const nbr_t& value) {
    nbr_t* cur = adj_lists_[src].end++;
    assert(adj_lists_[src].degree() <= capacity_[src]);
    *cur = value;
    return cur;
  }

  nbr_t* put_edge(vid_t src, nbr_t&& value) {
    nbr_t* cur = adj_lists_[src].end++;
    assert(adj_lists_[src].degree() <= capacity_[src]);
    *cur = std::move(value);
    return cur;
  }

  void dedup_neighbors(vid_t i) {
    int degree = adj_lists_[i].degree();
    if (degree > 0) {
      mutable_csr_impl::sort_neighbors(adj_lists_[i].begin, adj_lists_[i].end);
      adj_lists_[i].end = mutable_csr_impl::sorted_dedup(adj_lists_[i].begin,
                                                         adj_lists_[i].end);
    }
  }

  void sort_neighbors(vid_t i) {
    int degree = adj_lists_[i].degree();
    if (degree > 0) {
      mutable_csr_impl::sort_neighbors(adj_lists_[i].begin, adj_lists_[i].end);
    }
  }

  void dedup_neighbors_dense(const std::vector<int>& degree_to_add) {
    vid_t vnum = vertex_num();
    assert(vnum == degree_to_add.size());
    std::vector<nbr_t> buffer;
    for (vid_t i = 0; i < vnum; ++i) {
      int unsorted = degree_to_add[i];
      if (unsorted == 0) {
        continue;
      }
      int degree = adj_lists_[i].degree();
      bool minor_modification = (degree > (unsorted << 1));
      if (minor_modification) {
        auto new_begin = mutable_csr_impl::sort_neighbors_tail_dedup(
            adj_lists_[i].begin, adj_lists_[i].end, unsorted, buffer);
        if (new_begin != adj_lists_[i].begin) {
          int diff = new_begin - adj_lists_[i].begin;
          adj_lists_[i].begin = new_begin;
          capacity_[i] -= diff;
          vid_t prev = prev_[i];
          static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
          if (prev != sentinel) {
            capacity_[prev] += diff;
          }
        }
      } else {
        mutable_csr_impl::sort_neighbors(adj_lists_[i].begin,
                                         adj_lists_[i].end);
        adj_lists_[i].end = mutable_csr_impl::sorted_dedup(adj_lists_[i].begin,
                                                           adj_lists_[i].end);
      }
    }
  }

  void dedup_neighbors_sparse(const std::map<vid_t, int>& degree_to_add) {
    std::vector<nbr_t> buffer;
    for (auto& pair : degree_to_add) {
      vid_t i = pair.first;
      int unsorted = pair.second;
      int degree = adj_lists_[i].degree();
      bool minor_modification = (degree > (unsorted << 1));
      if (minor_modification) {
        auto new_begin = mutable_csr_impl::sort_neighbors_tail_dedup(
            adj_lists_[i].begin, adj_lists_[i].end, unsorted, buffer);
        if (new_begin != adj_lists_[i].begin) {
          int diff = new_begin - adj_lists_[i].begin;
          adj_lists_[i].begin = new_begin;
          capacity_[i] -= diff;
          vid_t prev = prev_[i];
          static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
          if (prev != sentinel) {
            capacity_[prev] += diff;
          }
        }
      } else {
        mutable_csr_impl::sort_neighbors(adj_lists_[i].begin,
                                         adj_lists_[i].end);
        adj_lists_[i].end = mutable_csr_impl::sorted_dedup(adj_lists_[i].begin,
                                                           adj_lists_[i].end);
      }
    }
  }

  void sort_neighbors_dense(const std::vector<int>& degree_to_add) {
    vid_t vnum = vertex_num();
    assert(vnum == degree_to_add.size());
    std::vector<nbr_t> buffer;
    for (vid_t i = 0; i < vnum; ++i) {
      int unsorted = degree_to_add[i];
      if (unsorted == 0) {
        continue;
      }
      int degree = adj_lists_[i].degree();
      bool minor_modification = (degree > (unsorted << 1));
      if (minor_modification) {
        mutable_csr_impl::sort_neighbors_tail(
            adj_lists_[i].begin, adj_lists_[i].end, unsorted, buffer);
      } else {
        mutable_csr_impl::sort_neighbors(adj_lists_[i].begin,
                                         adj_lists_[i].end);
      }
    }
  }

  void sort_neighbors_sparse(const std::map<vid_t, int>& degree_to_add) {
    std::vector<nbr_t> buffer;
    for (auto& pair : degree_to_add) {
      vid_t i = pair.first;
      int unsorted = pair.second;
      int degree = adj_lists_[i].degree();
      bool minor_modification = (degree > (unsorted << 1));
      if (minor_modification) {
        mutable_csr_impl::sort_neighbors_tail(
            adj_lists_[i].begin, adj_lists_[i].end, unsorted, buffer);
      } else {
        mutable_csr_impl::sort_neighbors(adj_lists_[i].begin,
                                         adj_lists_[i].end);
      }
    }
  }

  bool remove_one_with_tomb(vid_t src, vid_t dst) {
    return mutable_csr_impl::binary_remove_one_with_tomb(
        adj_lists_[src].begin, adj_lists_[src].end, dst);
  }

  bool remove_with_tomb(vid_t src, vid_t dst) {
    return mutable_csr_impl::binary_remove_with_tomb(adj_lists_[src].begin,
                                                     adj_lists_[src].end, dst);
  }

  void remove_vertex(vid_t i) { adj_lists_[i].end = adj_lists_[i].begin; }

  void shrink(vid_t i, size_t delta) {
    delta = std::min(delta, static_cast<size_t>(adj_lists_[i].degree()));
    adj_lists_[i].end -= delta;
  }

  void update_one(vid_t src, vid_t dst, const EDATA_T& value) {
    mutable_csr_impl::binary_update_one(adj_lists_[src].begin,
                                        adj_lists_[src].end, dst, value);
  }

  void update(vid_t src, vid_t dst, const EDATA_T& value) {
    mutable_csr_impl::binary_update(adj_lists_[src].begin, adj_lists_[src].end,
                                    dst, value);
  }

  template <typename FUNC_T>
  bool remove_one_with_tomb_if(const FUNC_T& func) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    vid_t vnum = vertex_num();
    for (vid_t i = 0; i < vnum; ++i) {
      auto begin = adj_lists_[i].begin;
      auto end = adj_lists_[i].end;
      while (begin != end) {
        if (func(i, *begin)) {
          begin->neighbor.SetValue(sentinel);
          return true;
        }
        ++begin;
      }
    }
  }

  template <typename FUNC_T>
  vid_t remove_with_tomb_if(const FUNC_T& func) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    vid_t vnum = vertex_num();
    vid_t ret = 0;
    for (vid_t i = 0; i < vnum; ++i) {
      auto begin = adj_lists_[i].begin;
      auto end = adj_lists_[i].end;
      while (begin != end) {
        if (func(i, *begin)) {
          begin->neighbor.SetValue(sentinel);
          ++ret;
        }
        ++begin;
      }
    }
    return ret;
  }

  template <typename FUNC_T>
  void remove_if(const FUNC_T& func) {
    vid_t vnum = vertex_num();
    for (vid_t i = 0; i < vnum; ++i) {
      auto begin = adj_lists_[i].begin;
      auto last = begin;
      auto end = adj_lists_[i].end;
      while (begin != end) {
        if (func(i, *begin)) {
          ++begin;
        } else {
          *last = std::move(*begin);
          ++begin;
          ++last;
        }
      }
      adj_lists_[i].end = last;
    }
  }

  void remove_tombs(vid_t i) {
    adj_lists_[i].end =
        mutable_csr_impl::remove_tombs(adj_lists_[i].begin, adj_lists_[i].end);
  }

  void clear_edges() {
    vid_t vnum = vertex_num();
    capacity_.clear();
    adj_lists_.clear();
    buffers_.clear();
    capacity_.resize(vnum, 0);
    adj_lists_.resize(vnum);
    mutable_csr_impl::Blob<vid_t, nbr_t> blob;
    buffers_.emplace_back(std::move(blob));
    nbr_t* ptr = buffers_[0].data();
    for (vid_t i = 0; i < vnum; ++i) {
      adj_lists_[i].begin = ptr;
      adj_lists_[i].end = ptr;
    }
    return;
  }

  template <typename IOADAPTOR_T>
  void Serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    vid_t vnum = vertex_num();
    std::vector<int> degree(vnum);
    size_t edge_num = 0;
    size_t total_capacity = 0;
    for (vid_t i = 0; i < vnum; ++i) {
      degree[i] = adj_lists_[i].degree();
      edge_num += degree[i];
      total_capacity += degree[i] * relax_rate;
    }

    InArchive ia;
    ia << vnum << edge_num << total_capacity;
    CHECK(writer->WriteArchive(ia));
    ia.Clear();

    if (vnum > 0) {
      CHECK(writer->Write(&degree[0], sizeof(int) * vnum));
    }

    if (std::is_pod<nbr_t>::value) {
      for (vid_t i = 0; i < vnum; ++i) {
        CHECK(writer->Write(adj_lists_[i].begin,
                            adj_lists_[i].degree() * sizeof(nbr_t)));
      }
    } else {
      for (vid_t i = 0; i < vnum; ++i) {
        auto ptr = adj_lists_[i].begin;
        auto end = adj_lists_[i].end;
        while (ptr != end) {
          ia << *ptr;
          ++ptr;
        }
      }
      CHECK(writer->WriteArchive(ia));
      ia.Clear();
    }
  }

  template <typename IOADAPTOR_T>
  void Deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    OutArchive oa;
    CHECK(reader->ReadArchive(oa));
    vid_t vnum;
    size_t edge_num, total_capacity;
    oa >> vnum >> edge_num >> total_capacity;
    oa.Clear();

    capacity_.resize(vnum);
    prev_.resize(vnum);
    next_.resize(vnum);
    adj_lists_.resize(vnum);

    std::vector<int> degree_list(vnum);
    if (vnum) {
      CHECK(reader->Read(&degree_list[0], sizeof(int) * vnum));
    }

    mutable_csr_impl::Blob<vid_t, nbr_t> buffer(total_capacity);

    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    nbr_t* ptr = buffer.data();
    for (vid_t i = 0; i < vnum; ++i) {
      prev_[i] = i - 1;
      next_[i] = i + 1;
      capacity_[i] = degree_list[i] * relax_rate;
      adj_lists_[i].begin = ptr;
      adj_lists_[i].end = ptr + degree_list[i];
      ptr += capacity_[i];
    }
    prev_[0] = sentinel;
    next_[vnum - 1] = sentinel;
    if (std::is_pod<nbr_t>::value) {
      for (vid_t i = 0; i < vnum; ++i) {
        CHECK(reader->Read(ptr, sizeof(nbr_t) * degree_list[i]));
      }
    } else {
      CHECK(reader->ReadArchive(oa));
      for (vid_t i = 0; i < vnum; ++i) {
        nbr_t* begin = adj_lists_[i].begin;
        nbr_t* end = adj_lists_[i].end;
        while (begin != end) {
          oa >> *begin;
          ++begin;
        }
      }
      oa.Clear();
    }
    buffers_.emplace_back(std::move(buffer));
  }

 private:
  void remove_node(vid_t i) {
    static constexpr vid_t sentinel = std::numeric_limits<vid_t>::max();
    vid_t prev = prev_[i];
    vid_t next = next_[i];
    if (next == sentinel && prev == sentinel) {
      return;
    }
    if (prev != sentinel) {
      capacity_[prev] += capacity_[i];
      next_[prev] = next;
    }
    if (next != sentinel) {
      prev_[next] = prev;
    }
  }

  template <typename _VID_T, typename _NBR_T>
  friend class MutableCSRBuilder;

  std::vector<int> capacity_;
  std::vector<vid_t> prev_, next_;
  std::vector<adj_list_t> adj_lists_;
  std::vector<mutable_csr_impl::Blob<vid_t, nbr_t>> buffers_;
};

}  // namespace grape

#endif  // GRAPE_GRAPH_MUTABLE_CSR_H_
