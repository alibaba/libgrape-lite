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

#ifndef GRAPE_GRAPH_IMMUTABLE_CSR_H_
#define GRAPE_GRAPH_IMMUTABLE_CSR_H_

#include <algorithm>
#include <vector>

#include <glog/logging.h>

#include "grape/config.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/utils/gcontainer.h"

namespace grape {

template <typename VID_T, typename NBR_T>
class ImmutableCSR;

template <typename VID_T, typename NBR_T>
class ImmutableCSRBuild {
  using vid_t = VID_T;
  using nbr_t = NBR_T;

 public:
  using vertex_range_t = VertexRange<VID_T>;

  ImmutableCSRBuild() {}
  ~ImmutableCSRBuild() {}

  void init(VID_T vnum) {
    vnum_ = vnum;
    degree_.clear();
    degree_.resize(vnum, 0);
  }

  void init(const VertexRange<VID_T>& range) {
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

  void build_offsets() {
    edge_num_ = 0;
    for (auto d : degree_) {
      edge_num_ += d;
    }
    edges_.clear();
    edges_.resize(edge_num_);
    offsets_.clear();
    offsets_.resize(vnum_ + 1);
    offsets_[0] = edges_.data();
    for (VID_T i = 0; i < vnum_; ++i) {
      offsets_[i + 1] = offsets_[i] + degree_[i];
    }
    CHECK_EQ(offsets_[vnum_], edges_.data() + edge_num_);
    {
      std::vector<int> tmp;
      tmp.swap(degree_);
    }
    iter_ = offsets_;
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
      std::sort(offsets_[i], offsets_[i + 1], func);
    }
  }

  void finish(ImmutableCSR<VID_T, NBR_T>& ret) {
    for (VID_T i = 0; i < vnum_; ++i) {
      std::sort(offsets_[i], offsets_[i + 1]);
    }

    ret.edges_.swap(edges_);
    ret.offsets_.swap(offsets_);
  }

 private:
  VID_T vnum_;
  size_t edge_num_;

  Array<nbr_t, Allocator<nbr_t>> edges_;
  Array<nbr_t*, Allocator<nbr_t*>> offsets_;
  std::vector<int> degree_;
  Array<nbr_t*, Allocator<nbr_t*>> iter_;
};

template <typename VID_T, typename NBR_T>
class ImmutableCSRStreamBuilder {
 public:
  template <typename ITER_T>
  void add_edges(const ITER_T& begin, const ITER_T& end) {
    ITER_T iter = begin;
    int degree = 0;
    while (iter != end) {
      edges_.push_back(*iter);
      ++iter;
      ++degree;
    }
    degree_.push_back(degree);
  }

  void finish(ImmutableCSR<VID_T, NBR_T>& ret) {
    ret.edges_.clear();
    ret.edges_.resize(edges_.size());
    std::copy(edges_.begin(), edges_.end(), ret.edges_.begin());
    ret.offsets_.clear();
    VID_T vnum = degree_.size();
    ret.offsets_.resize(vnum + 1);
    ret.offsets_[0] = ret.edges_.data();
    for (VID_T i = 0; i < vnum; ++i) {
      ret.offsets_[i + 1] = ret.offsets_[i] + degree_[i];
    }
  }

 private:
  std::vector<int> degree_;
  std::vector<NBR_T> edges_;
};

template <typename VID_T, typename NBR_T>
class ImmutableCSR {
 public:
  using vid_t = VID_T;
  using nbr_t = NBR_T;

  ImmutableCSR() {
    offsets_.resize(1);
    offsets_[0] = NULL;
  }

  ~ImmutableCSR() {}

  VID_T vertex_num() const { return offsets_.size() - 1; }

  bool empty() const { return offsets_.size() <= 1; }

  size_t edge_num() const { return edges_.size(); }

  int degree(VID_T i) const { return offsets_[i + 1] - offsets_[i]; }

  bool is_empty(VID_T i) const { return offsets_[i + 1] == offsets_[i]; }

  nbr_t* get_begin(VID_T i) { return offsets_[i]; }
  const nbr_t* get_begin(VID_T i) const { return offsets_[i]; }

  nbr_t* get_end(VID_T i) { return offsets_[i + 1]; }
  const nbr_t* get_end(VID_T i) const { return offsets_[i + 1]; }

  Array<nbr_t, Allocator<nbr_t>> const& get_edges() { return edges_; }
  Array<nbr_t*, Allocator<nbr_t*>> const& get_offsets() { return offsets_; }

  Array<nbr_t, Allocator<nbr_t>>& get_edges_mut() { return edges_; }
  Array<nbr_t*, Allocator<nbr_t*>>& get_offsets_mut() { return offsets_; }

  template <typename IOADAPTOR_T>
  void Serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    vid_t vnum = vertex_num();
    std::vector<int> degree(vnum);
    for (VID_T i = 0; i < vnum; ++i) {
      degree[i] = offsets_[i + 1] - offsets_[i];
    }

    InArchive ia;
    ia << vnum << edge_num();
    CHECK(writer->WriteArchive(ia));
    ia.Clear();

    if (vnum > 0) {
      CHECK(writer->Write(&degree[0], sizeof(int) * vnum));
    }

    if (std::is_pod<nbr_t>::value) {
      if (edge_num() > 0) {
        CHECK(writer->Write(&edges_[0], edge_num() * sizeof(nbr_t)));
      }
    } else {
      ia << edges_;
      CHECK(writer->WriteArchive(ia));
      ia.Clear();
    }
  }

  template <typename IOADAPTOR_T>
  void Deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    OutArchive oa;
    CHECK(reader->ReadArchive(oa));
    vid_t vnum;
    size_t edge_num;
    oa >> vnum >> edge_num;
    oa.Clear();

    edges_.clear();
    edges_.resize(edge_num);
    if (vnum > 0) {
      std::vector<int> degree(vnum);
      CHECK(reader->Read(&degree[0], sizeof(int) * vnum));
      offsets_.clear();
      offsets_.resize(vnum + 1);
      offsets_[0] = edges_.data();
      for (vid_t i = 0; i < vnum; ++i) {
        offsets_[i + 1] = offsets_[i] + degree[i];
      }
    }
    if (std::is_pod<nbr_t>::value) {
      CHECK(reader->Read(&edges_[0], sizeof(nbr_t) * edge_num));
    } else {
      CHECK(reader->ReadArchive(oa));
      oa >> edges_;
      oa.Clear();
    }
  }

 private:
  Array<nbr_t, Allocator<nbr_t>> edges_;
  Array<nbr_t*, Allocator<nbr_t*>> offsets_;

  template <typename _VID_T, typename _EDATA_T>
  friend class ImmutableCSRBuild;

  template <typename _VID_T, typename _EDATA_T>
  friend class ImmutableCSRStreamBuilder;
};

}  // namespace grape

#endif  // GRAPE_GRAPH_IMMUTABLE_CSR_H_
