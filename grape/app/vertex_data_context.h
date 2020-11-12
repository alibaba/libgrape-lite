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

#ifndef GRAPE_APP_VERTEX_DATA_CONTEXT_H_
#define GRAPE_APP_VERTEX_DATA_CONTEXT_H_

#include "grape/app/context_base.h"
#include "grape/utils/vertex_array.h"

namespace grape {

template <typename FRAG_T, typename DATA_T>
class VertexDataContext : public ContextBase {
  using fragment_t = FRAG_T;
  using vertex_t = typename fragment_t::vertex_t;
  using vertex_array_t = typename fragment_t::template vertex_array_t<DATA_T>;

 public:
  using data_t = DATA_T;

  explicit VertexDataContext(const fragment_t& fragment,
                             bool including_outer = false)
      : fragment_(fragment) {
    if (including_outer) {
      data_.Init(fragment.Vertices());
    } else {
      data_.Init(fragment.InnerVertices());
    }
  }

  const fragment_t& fragment() { return fragment_; }

  inline vertex_array_t& data() { return data_; }

 private:
  const fragment_t& fragment_;
  vertex_array_t data_;
};

}  // namespace grape

#endif  // GRAPE_APP_VERTEX_DATA_CONTEXT_H_
