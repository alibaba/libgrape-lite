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

#ifndef EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_OPT_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_OPT_CONTEXT_H_

#include <grape/grape.h>

#include <vector>

namespace grape {
/**
 * @brief Context for the parallel version of CDLP.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
#ifdef GID_AS_LABEL
class CDLPOptContext
    : public VertexDataContext<FRAG_T, typename FRAG_T::vid_t> {
#else
class CDLPOptContext
    : public VertexDataContext<FRAG_T, typename FRAG_T::oid_t> {
#endif

 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

#ifdef GID_AS_LABEL
  using label_t = vid_t;
#else
  using label_t = oid_t;
#endif
  explicit CDLPOptContext(const FRAG_T& fragment)
#ifdef GID_AS_LABEL
      : VertexDataContext<FRAG_T, typename FRAG_T::vid_t>(fragment, true),
#else
      : VertexDataContext<FRAG_T, typename FRAG_T::oid_t>(fragment, true),
#endif
        labels(this->data()) {
  }

  void Init(ParallelMessageManagerOpt& messages, int max_round,
            double threshold = 0.002) {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    this->max_round = max_round;
    this->threshold = threshold;
    changed.Init(inner_vertices);
    potential_change.Init(inner_vertices);
    new_ilabels.Init(frag.Vertices());

    step = 0;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << labels[v] << std::endl;
    }
  }

  typename FRAG_T::template vertex_array_t<label_t>& labels;
  typename FRAG_T::template vertex_array_t<label_t> new_ilabels;
  DenseVertexSet<typename FRAG_T::inner_vertices_t> potential_change, changed;

  int step = 0;
  int max_round = 0;
  double threshold = 0;
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_OPT_CONTEXT_H_
