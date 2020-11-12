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

#ifndef GRAPE_APP_VOID_CONTEXT_H_
#define GRAPE_APP_VOID_CONTEXT_H_
#include "grape/app/context_base.h"

namespace grape {

template <typename FRAG_T>
class VoidContext : public ContextBase {
  using fragment_t = FRAG_T;
  using vertex_t = typename fragment_t::vertex_t;

 public:
  explicit VoidContext(const fragment_t& fragment) : fragment_(fragment) {}

  const fragment_t& fragment() { return fragment_; }

 private:
  const fragment_t& fragment_;
};
}  // namespace grape
#endif  // GRAPE_APP_VOID_CONTEXT_H_
