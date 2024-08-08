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

#ifndef EXAMPLES_ANALYTICAL_APPS_KCORE_KCORE_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_KCORE_KCORE_CONTEXT_H_

namespace grape {

template <typename FRAG_T>
class KCoreContext : public VertexDataContext<FRAG_T, int> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit KCoreContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, int>(fragment, false),
        partial_result(this->data()) {}

  void Init(ParallelMessageManagerOpt& messages, int k_) { k = k_; }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    vid_t num = 0;
    for (auto v : inner_vertices) {
      if (partial_result[v] >= k) {
        ++num;
      }
    }

    LOG(INFO) << "[frag-" << frag.fid() << "] "
              << "KCore: " << num;
  }

  int k;
  typename FRAG_T::template inner_vertex_array_t<int>& partial_result;
  typename FRAG_T::template vertex_array_t<int> reduced_degrees;

  DenseVertexSet<typename FRAG_T::inner_vertices_t> curr_inner_updated,
      next_inner_updated;
  DenseVertexSet<typename FRAG_T::outer_vertices_t> outer_updated;

  DenseVertexSet<typename FRAG_T::inner_vertices_t> removed;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_KCORE_KCORE_CONTEXT_H_
