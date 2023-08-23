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

#include "grape/app/vertex_data_context.h"

namespace grape {
/**
 * @brief Context for the parallel version of WCC.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class WCCContext : public VertexDataContext<FRAG_T, typename FRAG_T::vid_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit WCCContext()
      : VertexDataContext<FRAG_T, typename FRAG_T::vid_t>(true),
        comp_id(this->data()) {}

  void Init(ParallelMessageManager& messages) {
    auto& frag = *this->fragment();

    curr_modified.Init(frag.Vertices());
    next_modified.Init(frag.Vertices());
  }

  typename FRAG_T::template vertex_array_t<vid_t>& comp_id;

  DenseVertexSet<vid_t> curr_modified, next_modified;

#ifdef PROFILING
  double preprocess_time = 0;
  double eval_time = 0;
  double postprocess_time = 0;
#endif
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_WCC_WCC_CONTEXT_H_
