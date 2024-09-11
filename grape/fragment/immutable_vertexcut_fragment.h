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

#include <glog/logging.h>

namespace grape {

template <typename OID_T, typename VID_T, typename VDATA_T, typename EDATA_T>
class ImmutableVertexcutFragment {};

template <typename VID_T>
class ImmutableVertexcutFragment<int64_t, VID_T, EmptyType, EmptyType> {
 public:
  using oid_t = int64_t;
  using vid_t = VID_T;
  using edata_t = EmptyType;
  using vdata_t = EmptyType;
  using vertices_t = VertexRange<oid_t>;
  using both_vertices_t = DualVertexRange<oid_t>;
  using edge_t = Edge<oid_t, edata_t>;

  template <typename T>
  using vertex_array_t = VertexArray<vertices_t, T>;

  template <typename T>
  using both_vertex_array_t = VertexArray<both_vertices_t, T>;

  ImmutableVertexcutFragment() = default;
  ~ImmutableVertexcutFragment() = default;

  void Init(const CommSpec& comm_spec, int64_t vnum,
            std::vector<edge_t>&& edges) {
    fid_ = comm_spec.fid();
    fnum_ = comm_spec.fnum();

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

    edges_ = std::move(edges);
  }

  const vertices_t& GetSourceVertices() const { return src_vertices_; }
  const vertices_t& GetDestinationVertices() const { return dst_vertices_; }
  const both_vertices_t& GetVertices() const { return vertices_; }

  const vertices_t& GetMasterVertices() const { return master_vertices_; }

  const std::vector<edge_t>& GetEdges() const { return edges_; }

  fid_t fid() const { return fid_; }
  fid_t fnum() const { return fnum_; }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), kSerializationFilenameFormat, prefix.c_str(),
             fid_);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    InArchive arc;
    arc << fid_ << fnum_ << edges_.size();
    arc << src_vertices_ << dst_vertices_ << vertices_ << master_vertices_;
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

    OutArchive arc;
    size_t edge_num;
    CHECK(io_adaptor->ReadArchive(arc));
    arc >> fid_ >> fnum_ >> edge_num;
    CHECK_EQ(fid_, comm_spec.fid());
    CHECK_EQ(fnum_, comm_spec.fnum());
    arc >> src_vertices_ >> dst_vertices_ >> vertices_ >> master_vertices_;

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
  }

  int64_t GetTotalVerticesNum() const {
    return partitioner_.get_total_vertices_num();
  }

  const VCPartitioner<int64_t>& GetPartitioner() const { return partitioner_; }

 private:
  fid_t fid_;
  fid_t fnum_;
  VCPartitioner<int64_t> partitioner_;
  vertices_t src_vertices_;
  vertices_t dst_vertices_;
  both_vertices_t vertices_;
  vertices_t master_vertices_;
  std::vector<edge_t> edges_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_IMMUTABLE_VERTEXCUT_FRAGMENT_H_
