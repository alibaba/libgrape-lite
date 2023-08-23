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
#include <unordered_map>

#include "grape/app/context_base.h"
#include "grape/utils/vertex_array.h"

namespace grape {

template <typename FRAG_T, typename DATA_T>
class VertexDataContext : public ContextBase<FRAG_T> {
  using fragment_t = FRAG_T;
  using vertex_t = typename fragment_t::vertex_t;
  using vertex_array_t = typename fragment_t::template vertex_array_t<DATA_T>;

 public:
  using data_t = DATA_T;

  explicit VertexDataContext(bool including_outer = false)
      : including_outer_(including_outer), initialized_(false) {}

  void set_fragment(const std::shared_ptr<fragment_t>& fragment) {
    if (initialized_) {
      auto iv = fragment->InnerVertices();
      vertex_array_t backup;

      backup.Init(iv);
      for (auto v : iv) {
        backup[v] = data_[v];
      }
      if (including_outer_) {
        data_.Init(fragment->Vertices(), 0);
      } else {
        data_.Init(fragment->InnerVertices(), 0);
      }
      for (auto v : iv) {
        data_[v] = backup[v];
      }
      CHECK_EQ(iv_size_, iv.size());
    } else {
      if (including_outer_) {
        data_.Init(fragment->Vertices(), 0);
      } else {
        data_.Init(fragment->InnerVertices(), 0);
      }
      initialized_ = true;
      iv_size_ = fragment->InnerVertices().size();
    }
    fragment_ = fragment;
  }

  std::shared_ptr<fragment_t>& fragment() { return fragment_; }

  inline vertex_array_t& data() { return data_; }

 private:
  bool including_outer_;
  bool initialized_;
  size_t iv_size_;
  std::shared_ptr<fragment_t> fragment_;
  vertex_array_t data_;
};

}  // namespace grape

#endif  // GRAPE_APP_VERTEX_DATA_CONTEXT_H_
