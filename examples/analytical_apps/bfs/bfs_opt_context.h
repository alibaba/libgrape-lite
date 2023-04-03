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

#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_OPT_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_OPT_CONTEXT_H_

#include <grape/grape.h>

#include <limits>

namespace grape {
/**
 * @brief Context for the parallel version of BFS.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class BFSOptContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using depth_type = int64_t;
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit BFSOptContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, int64_t>(fragment, true),
        partial_result(this->data()) {}

  void Init(ParallelMessageManagerOpt& messages, oid_t src_id) {
    source_id = src_id;
    partial_result.SetValue(std::numeric_limits<depth_type>::max());
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << partial_result[v] << std::endl;
    }
#ifdef PROFILING
    VLOG(2) << "preprocess_time: " << preprocess_time << "s.";
    VLOG(2) << "exec_time: " << exec_time << "s.";
    VLOG(2) << "postprocess_time: " << postprocess_time << "s.";
#endif
  }

  oid_t source_id;
  typename FRAG_T::template vertex_array_t<depth_type>& partial_result;
  DenseVertexSet<typename FRAG_T::inner_vertices_t> curr_inner_updated,
      next_inner_updated;

  depth_type current_depth = 0;
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_OPT_CONTEXT_H_
