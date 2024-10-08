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

#include "grape/grape.h"
#include "grape/utils/memory_tracker.h"

#include <iomanip>

namespace grape {

template <typename FRAG_T>
class PageRankVCContext : public VertexDataContext<FRAG_T, double> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  explicit PageRankVCContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, double>(fragment),
        master_result(this->data()) {
    curr_result.Init(fragment.Vertices());
    next_result.Init(fragment.Vertices());
    master_degree.Init(fragment.MasterVertices());

#ifdef TRACKING_MEMORY
    MemoryTracker::GetInstance().allocate(fragment.Vertices().size() *
                                          sizeof(double));
    MemoryTracker::GetInstance().allocate(fragment.Vertices().size() *
                                          sizeof(double));
    MemoryTracker::GetInstance().allocate(fragment.MasterVertices().size() *
                                          sizeof(double));
    MemoryTracker::GetInstance().allocate(fragment.MasterVertices().size() *
                                          sizeof(int));
#endif
  }

  void Init(GatherScatterMessageManager& messages, double delta,
            int max_round) {
    this->delta = delta;
    this->max_round = max_round;
    step = 0;
  }

  void Output(std::ostream& os) {
    auto& frag = this->fragment();
    auto master_vertices = frag.MasterVertices();
    for (auto v : master_vertices) {
      os << v.GetValue() << " " << std::scientific << std::setprecision(15)
         << master_result[v] << std::endl;
    }

#ifdef PROFILING
    VLOG(2) << "[frag-" << frag.fid() << "]: init degree: " << t0 << " s, "
            << "calc master result: " << t1 << " s, "
            << "propagate: " << t2 << " s";
#endif
  }

  typename FRAG_T::template both_vertex_array_t<double> curr_result;
  typename FRAG_T::template both_vertex_array_t<double> next_result;
  typename FRAG_T::template vertex_array_t<double>& master_result;
  typename FRAG_T::template vertex_array_t<int> master_degree;

  int64_t total_dangling_vnum = 0;
  int64_t graph_vnum;
  int step = 0;
  int max_round = 0;
  double delta = 0;

  double dangling_sum = 0.0;

#ifdef PROFILING
  double t0 = 0;  // init degree
  double t1 = 0;  // calc master result
  double t2 = 0;  // propogate
#endif
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_VC_CONTEXT_H_
