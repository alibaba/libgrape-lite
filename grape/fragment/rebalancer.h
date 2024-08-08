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

#ifndef GRAPE_FRAGMENT_REBALANCER_H_
#define GRAPE_FRAGMENT_REBALANCER_H_

#include <memory>

#include "grape/types.h"
#include "grape/vertex_map/vertex_map_beta.h"

namespace grape {

template <typename OID_T, typename VID_T>
class Rebalancer {
  using internal_oid_t = typename InternalOID<OID_T>::type;
  using vid_t = VID_T;

 public:
  Rebalancer(int vertex_factor,
             std::unique_ptr<VertexMap<OID_T, VID_T>>&& vertex_map)
      : vertex_factor_(vertex_factor), vertex_map_(std::move(vertex_map)) {
    fid_t fnum = vertex_map_->GetFragmentNum();
    id_parser_.init(fnum);
    degree_.resize(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      degree_[i].resize(vertex_map_->GetInnerVertexSize(i), 0);
    }
  }

  void inc_degree(const OID_T& oid) {
    VID_T gid;
    if (vertex_map_->GetGid(oid, gid)) {
      fid_t fid = id_parser_.get_fragment_id(gid);
      vid_t lid = id_parser_.get_local_id(gid);
      ++degree_[fid][lid];
    }
  }

  void finish(const CommSpec& comm_spec,
              VertexMap<OID_T, VID_T>& new_vertex_map) {
    for (auto& deg : degree_) {
      MPI_Allreduce(MPI_IN_PLACE, deg.data(), deg.size(), MPI_INT, MPI_SUM,
                    comm_spec.comm());
    }
    size_t total_edge_num = 0;
    size_t total_vertex_num = 0;
    for (auto& vec : degree_) {
      total_vertex_num += vec.size();
      for (auto deg : vec) {
        total_edge_num += deg;
      }
    }

    fid_t fnum = vertex_map_->GetFragmentNum();
    size_t total_score = total_edge_num + total_vertex_num * vertex_factor_;
    size_t expected_score = (total_score + fnum - 1) / fnum;

    fid_t self_fid = comm_spec.fid();
    fid_t cur_fid = 0;
    size_t cur_score = 0;
    std::vector<OID_T> native_oids;
    std::unique_ptr<MapPartitioner<OID_T>> new_partitioner(
        new MapPartitioner<OID_T>(fnum));
    for (fid_t i = 0; i < fnum; ++i) {
      vid_t vnum = vertex_map_->GetInnerVertexSize(i);
      for (vid_t j = 0; j < vnum; ++j) {
        OID_T cur_oid;
        CHECK(vertex_map_->GetOid(i, j, cur_oid));
        new_partitioner->SetPartitionId(internal_oid_t(cur_oid), cur_fid);
        if (cur_fid == self_fid) {
          native_oids.push_back(cur_oid);
        }

        size_t v_score = degree_[i][j] + vertex_factor_;
        cur_score += v_score;
        if (cur_score > expected_score && cur_fid < (fnum - 1)) {
          ++cur_fid;
          cur_score = 0;
        }
      }
    }
    CHECK_LE(cur_fid, fnum);

    VertexMapBuilder<OID_T, VID_T> builder(
        self_fid, fnum, std::move(new_partitioner), true, false);
    for (auto& oid : native_oids) {
      builder.add_vertex(oid);
    }
    builder.finish(comm_spec, new_vertex_map);
  }

 private:
  int vertex_factor_;
  std::unique_ptr<VertexMap<OID_T, VID_T>> vertex_map_;
  IdParser<VID_T> id_parser_;

  std::vector<std::vector<int>> degree_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_REBALANCER_H_
