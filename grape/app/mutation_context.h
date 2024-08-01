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

#ifndef GRAPE_APP_MUTATION_CONTEXT_H_
#define GRAPE_APP_MUTATION_CONTEXT_H_

#include <grape/config.h>
#include <grape/fragment/basic_fragment_mutator.h>
#include "grape/app/context_base.h"

namespace grape {

template <typename FRAG_T>
class MutationContext : public ContextBase {
  using fragment_t = FRAG_T;
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;
  using vdata_t = typename FRAG_T::vdata_t;
  using edata_t = typename FRAG_T::edata_t;
  using vertex_map_t = typename FRAG_T::vertex_map_t;
  using partitioner_t = typename vertex_map_t::partitioner_t;
  using vertex_t = typename FRAG_T::vertex_t;

  using oid_list = typename ShuffleBuffer<oid_t>::type;
  using vid_list = typename ShuffleBuffer<vid_t>::type;
  using vdata_list = typename ShuffleBuffer<vdata_t>::type;
  using edata_list = typename ShuffleBuffer<edata_t>::type;

 public:
  explicit MutationContext(const fragment_t& fragment)
      : fragment_(fragment),
        vm_ptr_(fragment.GetVertexMap()),
        partitioner_(vm_ptr_->GetPartitioner()) {
    fid_t fnum = fragment_->fnum();
    id_to_add_.resize(fnum);
    vdata_to_add_.resize(fnum);

    esrc_to_add_.resize(fnum);
    edst_to_add_.resize(fnum);
    edata_to_add_.resize(fnum);

    id_to_update_.resize(fnum);
    vdata_to_update_.resize(fnum);

    esrc_to_update_.resize(fnum);
    edst_to_update_.resize(fnum);
    edata_to_update_.resize(fnum);

    id_to_remove_.resize(fnum);

    esrc_to_remove_.resize(fnum);
    edst_to_remove_.resize(fnum);
  }

  void add_vertex(const oid_t& id, const vdata_t& data) {
    fid_t fid = partitioner_.GetPartitionId(id);
    id_to_add_[fid].push_back(id);
    vdata_to_add_[fid].push_back(data);
  }

  void add_edge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t src_fid = partitioner_.GetPartitionId(src);
    fid_t dst_fid = partitioner_.GetPartitionId(dst);
    esrc_to_add_[src_fid].push_back(src);
    edst_to_add_[src_fid].push_back(dst);
    edata_to_add_[src_fid].push_back(data);
    if (src_fid != dst_fid) {
      esrc_to_add_[dst_fid].push_back(src);
      edst_to_add_[dst_fid].push_back(dst);
      edata_to_add_[dst_fid].push_back(data);
    }
  }

  void add_edge(const vertex_t& src, const vertex_t& dst, const edata_t& data) {
    oid_t src_oid = fragment_.GetId(src);
    oid_t dst_oid = fragment_.GetId(dst);
    add_edge(src_oid, dst_oid, data);
  }

  void update_vertex(const oid_t& id, const vdata_t& data) {
    vid_t gid;
    if (vm_ptr_->GetGid(id, gid)) {
      parsed_vertices_to_update_.emplace_back(gid, data);
    } else {
      fid_t fid = partitioner_.GetPartitionId(id);
      id_to_update_[fid].push_back(id);
      vdata_to_update_[fid].push_back(data);
    }
  }

  void update_vertex(const vertex_t& v, const vdata_t& data) {
    parsed_vertices_to_update_.emplace_back(fragment_.Vertex2Gid(v), data);
  }

  void update_edge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t src_fid = partitioner_.GetPartitionId(src);
    fid_t dst_fid = partitioner_.GetPartitionId(dst);
    esrc_to_update_[src_fid].push_back(src);
    edst_to_update_[src_fid].push_back(dst);
    edata_to_update_[src_fid].push_back(data);
    if (src_fid != dst_fid) {
      esrc_to_update_[dst_fid].push_back(src);
      edst_to_update_[dst_fid].push_back(dst);
      edata_to_update_[dst_fid].push_back(data);
    }
  }

  void update_edge(const vertex_t& src, const vertex_t& dst,
                   const edata_t& data) {
    oid_t src_oid = fragment_.GetId(src);
    oid_t dst_oid = fragment_.GetId(dst);
    update_edge(src_oid, dst_oid, data);
  }

  void remove_vertex(const oid_t& id) {
    vid_t gid;
    if (vm_ptr_->GetGid(id, gid)) {
      parsed_vid_to_remove_.push_back(gid);
    } else {
      fid_t fid = partitioner_.GetPartitionId(id);
      id_to_remove_[fid].push_back(id);
    }
  }

  void remove_vertex(const vertex_t& v) {
    parsed_vid_to_remove_.push_back(fragment_.Vertex2Gid(v));
  }

  void remove_edge(const oid_t& src, const oid_t& dst) {
    fid_t src_fid = partitioner_.GetPartitionId(src);
    fid_t dst_fid = partitioner_.GetPartitionId(dst);
    esrc_to_remove_[src_fid].push_back(src);
    edst_to_remove_[src_fid].push_back(dst);
    if (src_fid != dst_fid) {
      esrc_to_remove_[dst_fid].push_back(src);
      edst_to_remove_[dst_fid].push_back(dst);
    }
  }

  void remove_edge(const vertex_t& src, const vertex_t& dst) {
    oid_t src_oid = fragment_.GetId(src);
    oid_t dst_oid = fragment_.GetId(dst);
    remove_edge(src_oid, dst_oid);
  }

  void apply_mutation(std::shared_ptr<fragment_t> fragment,
                      const CommSpec& comm_spec) {
    {
      CommSpec dup_comm_spec(comm_spec);
      dup_comm_spec.Dup();
      int local_to_mutate = 1;
      if (id_to_add_.empty() && esrc_to_add_.empty() &&
          parsed_vertices_to_update_.empty() && id_to_update_.empty() &&
          esrc_to_update_.empty() && parsed_vid_to_remove_.empty() &&
          id_to_remove_.empty() && esrc_to_remove_.empty()) {
        local_to_mutate = 0;
      }
      int global_to_mutate;
      MPI_Allreduce(&local_to_mutate, &global_to_mutate, 1, MPI_INT, MPI_SUM,
                    comm_spec.comm());
      if (global_to_mutate == 0) {
        return;
      }
    }
    BasicFragmentMutator<fragment_t> mutator(comm_spec, fragment);
    mutator.AddVerticesToRemove(std::move(parsed_vid_to_remove_));
    mutator.AddVerticesToUpdate(std::move(parsed_vertices_to_update_));
    mutator.Start();
    mutator.AddVertices(std::move(id_to_add_), std::move(vdata_to_add_));
    mutator.RemoveVertices(std::move(id_to_remove_));
    mutator.UpdateVertices(std::move(id_to_update_),
                           std::move(vdata_to_update_));
    mutator.AddEdges(std::move(esrc_to_add_), std::move(edst_to_add_),
                     std::move(edata_to_update_));
    mutator.RemoveEdges(std::move(esrc_to_remove_), std::move(edst_to_remove_));
    mutator.UpdateEdges(std::move(esrc_to_update_), std::move(edst_to_update_),
                        std::move(edata_to_update_));
    mutator.MutateFragment();
    id_to_add_.clear();
    vdata_to_add_.clear();
    esrc_to_add_.clear();
    edst_to_add_.clear();
    edata_to_add_.clear();
    parsed_vertices_to_update_.clear();
    id_to_update_.clear();
    vdata_to_update_.clear();
    esrc_to_update_.clear();
    edst_to_update_.clear();
    edata_to_update_.clear();
    parsed_vid_to_remove_.clear();
    id_to_remove_.clear();
    esrc_to_remove_.clear();
    edst_to_remove_.clear();
  }

 private:
  const fragment_t& fragment_;
  const std::shared_ptr<vertex_map_t> vm_ptr_;
  const partitioner_t& partitioner_;

  std::vector<oid_list> id_to_add_;
  std::vector<vdata_list> vdata_to_add_;

  std::vector<oid_list> esrc_to_add_;
  std::vector<oid_list> edst_to_add_;
  std::vector<edata_list> edata_to_add_;

  std::vector<internal::Vertex<vid_t, vdata_t>> parsed_vertices_to_update_;
  std::vector<oid_list> id_to_update_;
  std::vector<vdata_list> vdata_to_update_;

  std::vector<oid_list> esrc_to_update_;
  std::vector<oid_list> edst_to_update_;
  std::vector<edata_list> edata_to_update_;

  std::vector<vid_t> parsed_vid_to_remove_;
  std::vector<oid_list> id_to_remove_;

  std::vector<oid_list> esrc_to_remove_;
  std::vector<oid_list> edst_to_remove_;
};

}  // namespace grape

#endif  // GRAPE_APP_MUTATION_CONTEXT_H_
