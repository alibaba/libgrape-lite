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

#ifndef GRAPE_APP_CONTEXT_BASE_H_
#define GRAPE_APP_CONTEXT_BASE_H_

#include <memory>
#include <ostream>

#include "grape/types.h"

namespace grape {

/**
 * @brief ContextBase is the base class for all user-defined contexts. A
 * context manages data through the whole computation. The data won't be cleared
 * during supersteps.
 *
 */
class ContextBase {
 public:
  ContextBase() = default;
  virtual ~ContextBase() = default;

  /**
   * @brief Output function to implement for result output.
   * @note: This pure virtual function works as an interface, instructing users
   * to implement in their defined context. The Output in the inherited apps
   * would be invoked directly, not via virtual functions.
   *
   * @param frag
   * @param os
   */
  virtual void Output(std::ostream& os) {}
};

}  // namespace grape

#endif  // GRAPE_APP_CONTEXT_BASE_H_
