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

#ifndef GRAPE_FRAGMENT_IMMUTABLE_VERTEXCUT_FRAGMENT_H_
#define GRAPE_FRAGMENT_IMMUTABLE_VERTEXCUT_FRAGMENT_H_

#include <atomic>

#include <glog/logging.h>

#include "grape/fragment/fragment_base.h"
#include "grape/graph/edge.h"
#include "grape/types.h"
#include "grape/utils/memory_inspector.h"
#include "grape/utils/vertex_array.h"
#include "grape/vertex_map/partitioner.h"
#include "grape/worker/comm_spec.h"

namespace grape {

// #define USE_EDGE_ARRAY

template <typename OID_T, typename VDATA_T, typename EDATA_T>
class ImmutableVertexcutFragment {};

template <>
class ImmutableVertexcutFragment<int64_t, EmptyType, EmptyType>
    : public FragmentBase<int64_t, EmptyType, EmptyType> {
 public:
  using oid_t = int64_t;
  using edata_t = EmptyType;
  using vdata_t = EmptyType;
  using vertices_t = VertexRange<oid_t>;
  using both_vertices_t = DualVertexRange<oid_t>;
  using edge_t = Edge<oid_t, edata_t>;
  using base_t = FragmentBase<oid_t, vdata_t, edata_t>;

  struct edge_bucket_t {
    edge_bucket_t() : begin_(nullptr), end_(nullptr) {}
    edge_bucket_t(const edge_t* begin, const edge_t* end)
        : begin_(begin), end_(end) {}

    const edge_t* begin() const { return begin_; }
    const edge_t* end() const { return end_; }

    const edge_t* begin_;
    const edge_t* end_;
  };

  static constexpr FragmentType fragment_type = FragmentType::kVertexCut;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  template <typename T>
  using vertex_array_t = VertexArray<vertices_t, T>;

  template <typename T>
  using both_vertex_array_t = VertexArray<both_vertices_t, T>;

  ImmutableVertexcutFragment() = default;
  ~ImmutableVertexcutFragment() {}

  using base_t::fid_;
  using base_t::fnum_;

  void Init(const CommSpec& comm_spec, int64_t vnum,
            std::vector<edge_t>&& edges, int bucket_num,
            std::vector<size_t>&& bucket_edge_offsets) {
    base_t::init(comm_spec.fid(), comm_spec.fnum(), false);

    partitioner_.init(comm_spec.fnum(), vnum);
    src_vertices_ = partitioner_.get_src_vertices(fid_);
    dst_vertices_ = partitioner_.get_dst_vertices(fid_);
    master_vertices_ = partitioner_.get_master_vertices(fid_);

    int64_t src_begin = src_vertices_.begin_value();
    int64_t src_end = src_vertices_.end_value();
    int64_t dst_begin = dst_vertices_.begin_value();
    int64_t dst_end = dst_vertices_.end_value();

    if (src_end < dst_begin) {
      vertices_ = both_vertices_t(src_begin, src_end, dst_begin, dst_end);
    } else if (dst_end < src_begin) {
      vertices_ = both_vertices_t(dst_begin, dst_end, src_begin, src_end);
    } else {
      vertices_ = both_vertices_t(0, 0, std::min(src_begin, dst_begin),
                                  std::max(src_end, dst_end));
    }

#ifdef USE_EDGE_ARRAY
    edges_.resize(edges.size());
    MemoryInspector::GetInstance().allocate(sizeof(edge_t) * edges_.size());
    std::copy(edges.begin(), edges.end(), edges_.begin());
#else
    edges_ = std::move(edges);
#endif
    bucket_num_ = bucket_num;
    bucket_edge_offsets_ = std::move(bucket_edge_offsets);
  }

  void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) override {}

  const vertices_t& SourceVertices() const { return src_vertices_; }
  const vertices_t& DestinationVertices() const { return dst_vertices_; }
  const both_vertices_t& Vertices() const { return vertices_; }

  const vertices_t& MasterVertices() const { return master_vertices_; }

#ifdef USE_EDGE_ARRAY
  const Array<edge_t, Allocator<edge_t>>& GetEdges() const { return edges_; }
#else
  const std::vector<edge_t>& GetEdges() const { return edges_; }
#endif

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid_);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    base_t::serialize(io_adaptor);

    InArchive arc;
    arc << edges_.size();
    arc << src_vertices_ << dst_vertices_ << vertices_ << master_vertices_;
    arc << bucket_num_ << bucket_edge_offsets_;
    CHECK(io_adaptor->WriteArchive(arc));
    arc.Clear();

    partitioner_.serialize(io_adaptor);

    if (std::is_pod<edata_t>::value && std::is_pod<oid_t>::value) {
      LOG(INFO) << "is pod";
      if (!edges_.empty()) {
        io_adaptor->Write(edges_.data(), edges_.size() * sizeof(edge_t));
      }
    } else {
      LOG(FATAL) << "is not pod";

      arc << edges_;
      if (!arc.Empty()) {
        CHECK(io_adaptor->WriteArchive(arc));
        arc.Clear();
      }
    }
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const CommSpec& comm_spec, const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             comm_spec.fid());
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    base_t::deserialize(io_adaptor);

    OutArchive arc;
    size_t edge_num;
    CHECK(io_adaptor->ReadArchive(arc));
    arc >> edge_num;
    CHECK_EQ(fid_, comm_spec.fid());
    CHECK_EQ(fnum_, comm_spec.fnum());
    arc >> src_vertices_ >> dst_vertices_ >> vertices_ >> master_vertices_;
    arc >> bucket_num_ >> bucket_edge_offsets_;

    partitioner_.deserialize(io_adaptor);

    if (std::is_pod<edata_t>::value && std::is_pod<oid_t>::value) {
      LOG(INFO) << "is pod";
      edges_.resize(edge_num);
      if (edge_num > 0) {
        CHECK(io_adaptor->Read(edges_.data(), edge_num * sizeof(edge_t)));
      }
    } else {
      LOG(FATAL) << "is not pod";
      arc.Clear();
      CHECK(io_adaptor->ReadArchive(arc));
      arc >> edges_;
    }

    MemoryInspector::GetInstance().allocate(sizeof(edge_t) * edges_.size());
  }

  size_t GetTotalVerticesNum() const override {
    return partitioner_.get_total_vertices_num();
  }

  size_t GetEdgeNum() const override { return edges_.size(); }

  size_t GetVerticesNum() const override { return vertices_.size(); }

  const VCPartitioner<int64_t>& GetPartitioner() const { return partitioner_; }

  edge_bucket_t GetEdgesOfBucket(int src_bucket_id, int dst_bucket_id) const {
    int idx = src_bucket_id * bucket_num_ + dst_bucket_id;
    if (static_cast<size_t>(idx) >= bucket_edge_offsets_.size()) {
      return edge_bucket_t();
    }
    return edge_bucket_t(edges_.data() + bucket_edge_offsets_[idx],
                         edges_.data() + bucket_edge_offsets_[idx + 1]);
  }

  int GetBucketNum() const { return bucket_num_; }

 private:
  VCPartitioner<int64_t> partitioner_;
  vertices_t src_vertices_;
  vertices_t dst_vertices_;
  both_vertices_t vertices_;
  vertices_t master_vertices_;
#ifdef USE_EDGE_ARRAY
  Array<edge_t, Allocator<edge_t>> edges_;
#else
  std::vector<edge_t> edges_;
#endif
  int bucket_num_;
  std::vector<size_t> bucket_edge_offsets_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_IMMUTABLE_VERTEXCUT_FRAGMENT_H_
