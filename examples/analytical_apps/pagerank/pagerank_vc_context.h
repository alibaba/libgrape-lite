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

#ifndef EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_CONTEXT_H_

namespace grape {

template <typename FRAG_T>
class PageRankVCContext {
  using oid_t = typename FRAG_T::oid_t;

 public:
  const FRAG_T& fragment() { return fragment_; }

  explicit PageRankVCContext(const FRAG_T& fragment) : fragment_(fragment) {
    curr_result.Init(fragment.GetVertices());
    next_result.Init(fragment.GetVertices());
    master_result.Init(fragment.GetMasterVertices());
    master_degree.Init(fragment.GetMasterVertices());
  }

  void Init(VCMessageManager& messages, double delta, int max_round) {
    this->delta = delta;
    this->max_round = max_round;
    step = 0;
  }

  void Output(std::ostream& os) {
    auto& frag = this->fragment();
    auto master_vertices = frag.GetMasterVertices();
    for (auto v : master_vertices) {
      os << v.GetValue() << " " << std::scientific << std::setprecision(15)
         << master_result[v] << std::endl;
    }
  }

  typename FRAG_T::template both_vertex_array_t<double> curr_result;
  typename FRAG_T::template both_vertex_array_t<double> next_result;
  typename FRAG_T::template vertex_array_t<double> master_result;
  typename FRAG_T::template vertex_array_t<int> master_degree;

  int64_t total_dangling_vnum = 0;
  int64_t graph_vnum;
  int step = 0;
  int max_round = 0;
  double delta = 0;

  double dangling_sum = 0.0;

  const FRAG_T& fragment_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_CONTEXT_H_