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

#ifndef GRAPE_APP_AUTO_APP_BASE_H_
#define GRAPE_APP_AUTO_APP_BASE_H_

#include <memory>

#include "grape/types.h"

namespace grape {

template <typename T>
class AutoParallelMessageManager;

template <typename T>
class AutoWorker;

/**
 * @brief AutoAppBase is a base class for auto-parallel apps.
 * It contains an AutoParallelMessageManager to process messages implicitly
 * during the computation.
 *
 * @tparam FRAG_T
 * @tparam CONTEXT_T
 */
template <typename FRAG_T, typename CONTEXT_T>
class AutoAppBase {
 public:
  static constexpr bool need_split_edges = false;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kSyncOnOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  using message_manager_t = AutoParallelMessageManager<FRAG_T>;
  AutoAppBase() = default;
  virtual ~AutoAppBase() = default;

  /**
   * @brief Partial evaluation to implement.
   * @note: This pure virtual function works as an interface, instructing users
   * to implement in the specific app. The PEval in the inherited apps would be
   * invoked directly, not via virtual functions.
   *
   * @param graph
   * @param context
   * @param messages
   */
  virtual void PEval(const FRAG_T& graph, CONTEXT_T& context) = 0;

  /**
   * @brief Incremental evaluation to implement.
   * @note: This pure virtual function works as an interface, instructing users
   * to implement in the specific app. The IncEval in the inherited apps would
   * be invoked directly, not via virtual functions.
   *
   * @param graph
   * @param context
   * @param messages
   */
  virtual void IncEval(const FRAG_T& graph, CONTEXT_T& context) = 0;
};

#define INSTALL_AUTO_WORKER(APP_T, CONTEXT_T, FRAG_T)                  \
 public:                                                               \
  using fragment_t = FRAG_T;                                           \
  using context_t = CONTEXT_T;                                         \
  using message_manager_t = grape::AutoParallelMessageManager<FRAG_T>; \
  using worker_t = grape::AutoWorker<APP_T>;                           \
  static std::shared_ptr<worker_t> CreateWorker(                       \
      std::shared_ptr<APP_T> app, std::shared_ptr<FRAG_T> frag) {      \
    return std::shared_ptr<worker_t>(new worker_t(app, frag));         \
  }

}  // namespace grape

#endif  // GRAPE_APP_AUTO_APP_BASE_H_
