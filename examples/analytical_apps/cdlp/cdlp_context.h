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

#ifndef EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_CONTEXT_H_

#include <vector>

#include <grape/grape.h>

namespace grape {
/**
 * @brief Context for the parallel version of CDLP.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class CDLPContext : public ContextBase<FRAG_T> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

#ifdef GID_AS_LABEL
  using label_t = vid_t;
#else
  using label_t = oid_t;
#endif
  void Init(const FRAG_T& frag, ParallelMessageManager& messages,
            int max_round) {
    this->max_round = max_round;
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();

    labels.Init(vertices);
    changed.Init(inner_vertices);

#ifdef PROFILING
    preprocess_time = 0;
    exec_time = 0;
    postprocess_time = 0;
#endif
    step = 0;
  }

  void Output(const FRAG_T& frag, std::ostream& os) {
    auto inner_vertices = frag.InnerVertices();
    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << labels[v] << std::endl;
    }
  }

  VertexArray<label_t, vid_t> labels;
  VertexArray<bool, vid_t> changed;

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif

  int step = 0;
  int max_round = 0;

#ifdef RANDOM_LABEL
  std::vector<std::mt19937> random_engines;
#endif
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_CDLP_CDLP_CONTEXT_H_
