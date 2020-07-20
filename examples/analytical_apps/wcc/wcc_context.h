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

#ifndef EXAMPLES_ANALYTICAL_APPS_WCC_WCC_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_WCC_WCC_CONTEXT_H_

#include <grape/grape.h>

namespace grape {
/**
 * @brief Context for the parallel version of WCC.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class WCCContext : public ContextBase<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  void Init(const FRAG_T& frag, ParallelMessageManager& messages) {
    auto vertices = frag.Vertices();

    comp_id.Init(vertices);

    curr_modified.init(frag.GetVerticesNum());
    next_modified.init(frag.GetVerticesNum());
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto inner_vertices = frag.InnerVertices();
    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << comp_id[v] << std::endl;
    }
#ifdef PROFILING
    VLOG(2) << "preprocess_time: " << preprocess_time << "s.";
    VLOG(2) << "eval_time: " << eval_time << "s.";
    VLOG(2) << "postprocess_time: " << postprocess_time << "s.";
#endif
  }

  VertexArray<vid_t, vid_t> comp_id;

  Bitset curr_modified, next_modified;

#ifdef PROFILING
  double preprocess_time = 0;
  double eval_time = 0;
  double postprocess_time = 0;
#endif
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_WCC_WCC_CONTEXT_H_
