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
  using vertex_t = typename FRAG_T::vertex_t;

 public:
  explicit MutationContext(const fragment_t& fragment)
      : fragment_(fragment), vm_ptr_(fragment.GetVertexMap()) {}

  void add_vertex(const oid_t& id, const vdata_t& data) {
    vid_to_add_.push_back(id);
    vdata_to_add_.push_back(data);
  }

  void add_edge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    esrc_to_add_.push_back(src);
    edst_to_add_.push_back(dst);
    edata_to_add_.push_back(data);
  }

  void add_edge(const vertex_t& src, const vertex_t& dst, const edata_t& data) {
    esrc_to_add_.push_back(fragment_.GetId(src));
    edst_to_add_.push_back(fragment_.GetId(dst));
    edata_to_add_.push_back(data);
  }

  void update_vertex(const oid_t& id, const vdata_t& data) {
    vid_t gid;
    if (vm_ptr_->GetGid(id, gid)) {
      vertices_to_update_.emplace_back(gid, data);
    }
  }

  void update_vertex(fid_t fid, const oid_t& id, const vdata_t& data) {
    vid_t gid;
    if (vm_ptr_->GetGid(fid, id, gid)) {
      vertices_to_update_.emplace_back(gid, data);
    }
  }

  void update_vertex(const vertex_t& v, const vdata_t& data) {
    vertices_to_update_.emplace_back(fragment_.Vertex2Gid(v), data);
  }

  void update_edge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    vid_t src_gid, dst_gid;
    if (vm_ptr_->GetGid(src, src_gid) && vm_ptr_->GetGid(dst, dst_gid)) {
      esrc_to_update_.push_back(src_gid);
      edst_to_update_.push_back(dst_gid);
      edata_to_update_.push_back(data);
    }
  }

  void update_edge(fid_t src_fid, const oid_t& src, fid_t dst_fid,
                   const oid_t& dst, const edata_t& data) {
    vid_t src_gid, dst_gid;
    if (vm_ptr_->GetGid(src_fid, src, src_gid) &&
        vm_ptr_->GetGid(dst_fid, dst, dst_gid)) {
      esrc_to_update_.push_back(src_gid);
      edst_to_update_.push_back(dst_gid);
      edata_to_update_.push_back(data);
    }
  }

  void update_edge(const vertex_t& src, const vertex_t& dst,
                   const edata_t& data) {
    esrc_to_update_.push_back(fragment_.Vertex2Gid(src));
    edst_to_update_.push_back(fragment_.Vertex2Gid(dst));
    edata_to_update_.push_back(data);
  }

  void remove_vertex(const oid_t& id) {
    vid_t gid;
    if (vm_ptr_->GetGid(id, gid)) {
      vid_to_remove_.push_back(gid);
    }
  }

  void remove_vertex(fid_t fid, const oid_t& id) {
    vid_t gid;
    if (vm_ptr_->GetGid(fid, id, gid)) {
      vid_to_remove_.push_back(gid);
    }
  }

  void remove_vertex(const vertex_t& v) {
    vid_to_remove_.push_back(fragment_.Vertex2Gid(v));
  }

  void remove_edge(const oid_t& src, const oid_t& dst) {
    vid_t src_gid, dst_gid;
    if (vm_ptr_->GetGid(src, src_gid) && vm_ptr_->GetGid(dst, dst_gid)) {
      esrc_to_remove_.push_back(src_gid);
      edst_to_remove_.push_back(dst_gid);
    }
  }

  void remove_edge(fid_t src_fid, const oid_t& src, fid_t dst_fid,
                   const oid_t& dst) {
    vid_t src_gid, dst_gid;
    if (vm_ptr_->GetGid(src_fid, src, src_gid) &&
        vm_ptr_->GetGid(dst_fid, dst, dst_gid)) {
      esrc_to_remove_.push_back(src_gid);
      edst_to_remove_.push_back(dst_gid);
    }
  }

  void remove_edge(const vertex_t& src, const vertex_t& dst) {
    esrc_to_remove_.push_back(fragment_.Vertex2Gid(src));
    edst_to_remove_.push_back(fragment_.Vertex2Gid(dst));
  }

  void apply_mutation(std::shared_ptr<fragment_t> fragment,
                      const CommSpec& comm_spec) {
    {
      CommSpec dup_comm_spec(comm_spec);
      dup_comm_spec.Dup();
      int local_to_mutate = 1;
      if (vid_to_add_.empty() && esrc_to_add_.empty() &&
          vid_to_remove_.empty() && esrc_to_remove_.empty() &&
          vertices_to_update_.empty() && esrc_to_update_.empty()) {
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
    mutator.Start();
    size_t add_v_num = vid_to_add_.size();
    for (size_t i = 0; i < add_v_num; ++i) {
      mutator.AddVertex(vid_to_add_[i], vdata_to_add_[i]);
    }
    size_t add_e_num = esrc_to_add_.size();
    for (size_t i = 0; i < add_e_num; ++i) {
      mutator.AddEdge(esrc_to_add_[i], edst_to_add_[i], edata_to_add_[i]);
    }
    mutator.UpdateVertexGidList(std::move(vertices_to_update_));
    size_t update_e_num = esrc_to_update_.size();
    for (size_t i = 0; i < update_e_num; ++i) {
      mutator.UpdateEdgeGid(esrc_to_update_[i], edst_to_update_[i],
                            edata_to_update_[i]);
    }
    mutator.RemoveVertexGidList(std::move(vid_to_remove_));
    size_t remove_e_num = esrc_to_remove_.size();
    for (size_t i = 0; i < remove_e_num; ++i) {
      mutator.RemoveEdgeGid(esrc_to_remove_[i], edst_to_remove_[i]);
    }
    mutator.MutateFragment();

    vid_to_add_.clear();
    vdata_to_add_.clear();
    esrc_to_add_.clear();
    edst_to_add_.clear();
    edata_to_add_.clear();
    vertices_to_update_.clear();
    esrc_to_update_.clear();
    edst_to_update_.clear();
    edata_to_update_.clear();
    vid_to_remove_.clear();
    esrc_to_remove_.clear();
    edst_to_remove_.clear();
  }

 private:
  const fragment_t& fragment_;
  const std::shared_ptr<vertex_map_t> vm_ptr_;

  std::vector<oid_t> vid_to_add_;
  std::vector<vdata_t> vdata_to_add_;

  std::vector<oid_t> esrc_to_add_;
  std::vector<oid_t> edst_to_add_;
  std::vector<edata_t> edata_to_add_;

  std::vector<internal::Vertex<vid_t, vdata_t>> vertices_to_update_;

  std::vector<vid_t> esrc_to_update_;
  std::vector<vid_t> edst_to_update_;
  std::vector<edata_t> edata_to_update_;

  std::vector<vid_t> vid_to_remove_;

  std::vector<vid_t> esrc_to_remove_;
  std::vector<vid_t> edst_to_remove_;
};

}  // namespace grape

#endif  // GRAPE_APP_MUTATION_CONTEXT_H_
