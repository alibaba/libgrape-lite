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

#ifndef EXAMPLES_ANALYTICAL_APPS_BFS_BFS_AUTO_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_BFS_BFS_AUTO_CONTEXT_H_

#include <limits>

#include <grape/grape.h>

namespace grape {
/**
 * @brief Context for the auto-parallel version of BFS.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class BFSAutoContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit BFSAutoContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, int64_t>(fragment),
        partial_result(this->data()) {}

  void Init(AutoParallelMessageManager<FRAG_T>& messages, oid_t src_id) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();

    source_id = src_id;
    partial_result.Init(vertices, std::numeric_limits<int64_t>::max(),
                        [](int64_t* lhs, int64_t rhs) {
                          if (*lhs > rhs) {
                            *lhs = rhs;
                            return true;
                          } else {
                            return false;
                          }
                        });

    messages.RegisterSyncBuffer(frag, &partial_result,
                                MessageStrategy::kSyncOnOuterVertex);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();

    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << partial_result[v] << std::endl;
    }
  }

  oid_t source_id;
  SyncBuffer<typename FRAG_T::vertices_t, int64_t> partial_result;
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_BFS_BFS_AUTO_CONTEXT_H_
