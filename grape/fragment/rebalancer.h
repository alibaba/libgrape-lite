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
#include "grape/vertex_map/vertex_map.h"

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
    fid_t fnum = vertex_map_->GetFragmentNum();
    fid_t self_fid = comm_spec.fid();
    for (auto& deg : degree_) {
      MPI_Allreduce(MPI_IN_PLACE, deg.data(), deg.size(), MPI_INT, MPI_SUM,
                    comm_spec.comm());
    }
    size_t total_score = 0;
    std::vector<size_t> frag_scores_before, frag_scores_after;
    for (auto& vec : degree_) {
      size_t cur_score = vec.size() * vertex_factor_;
      for (auto deg : vec) {
        cur_score += deg;
      }

      frag_scores_before.push_back(cur_score);
      total_score += cur_score;
    }
    size_t expected_score = (total_score + fnum - 1) / fnum;
    std::vector<OID_T> native_oids;
    std::unique_ptr<IPartitioner<OID_T>> new_partitioner(nullptr);
    if (vertex_map_->GetPartitioner().type() ==
        PartitionerType::kMapPartitioner) {
      fid_t cur_fid = 0;
      size_t cur_score = 0;

      new_partitioner = std::unique_ptr<MapPartitioner<OID_T>>(
          new MapPartitioner<OID_T>(fnum));
      frag_scores_after.resize(fnum, 0);
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
          frag_scores_after[cur_fid] += v_score;

          cur_score += v_score;
          if (cur_score > expected_score && cur_fid < (fnum - 1)) {
            ++cur_fid;
            cur_score = 0;
          }
        }
      }
      CHECK_LE(cur_fid, fnum);
    } else if (vertex_map_->GetPartitioner().type() ==
               PartitionerType::kSegmentedPartitioner) {
      size_t cur_score = 0;
      fid_t last_fid = 0;
      std::vector<OID_T> boundaries;
      frag_scores_after.resize(fnum, 0);
      for (fid_t i = 0; i < fnum; ++i) {
        std::vector<std::pair<OID_T, vid_t>> frag_vertices;
        vid_t vnum = vertex_map_->GetInnerVertexSize(i);
        frag_vertices.reserve(vnum);
        for (vid_t j = 0; j < vnum; ++j) {
          OID_T cur_oid;
          CHECK(vertex_map_->GetOid(i, j, cur_oid));
          frag_vertices.emplace_back(cur_oid, j);
        }
        std::sort(
            frag_vertices.begin(), frag_vertices.end(),
            [](const std::pair<OID_T, vid_t>& a,
               const std::pair<OID_T, vid_t>& b) { return a.first < b.first; });

        for (auto& pair : frag_vertices) {
          cur_score += (degree_[i][pair.second] + vertex_factor_);
          fid_t cur_fid = std::min<fid_t>(cur_score / expected_score, fnum - 1);
          frag_scores_after[cur_fid] +=
              (degree_[i][pair.second] + vertex_factor_);
          if (cur_fid != last_fid) {
            last_fid = cur_fid;
            boundaries.push_back(pair.first);
          }
          if (cur_fid == self_fid) {
            native_oids.emplace_back(pair.first);
          }
        }
      }
      CHECK_EQ(boundaries.size(), fnum - 1);
      new_partitioner = std::unique_ptr<SegmentedPartitioner<OID_T>>(
          new SegmentedPartitioner<OID_T>(boundaries));
    } else {
      LOG(FATAL) << "Unsupported partitioner type - "
                 << static_cast<int>(vertex_map_->GetPartitioner().type());
    }
    IdxerType idxer_type = vertex_map_->idxer_type();
    CHECK(idxer_type != IdxerType::kLocalIdxer)
        << "Rebalancer only supports global vertex map";
    VertexMapBuilder<OID_T, VID_T> builder(
        self_fid, fnum, std::move(new_partitioner), idxer_type);
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
