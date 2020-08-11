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

#ifndef GRAPE_WORKER_AUTO_WORKER_H_
#define GRAPE_WORKER_AUTO_WORKER_H_

#include <mpi.h>

#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>

#include "grape/communication/communicator.h"
#include "grape/config.h"
#include "grape/parallel/auto_parallel_message_manager.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/worker/comm_spec.h"

namespace grape {

template <typename FRAG_T, typename CONTEXT_T>
class AutoAppBase;

/**
 * @brief A Worker manages the computation cycle. AutoWorker is a kind of worker
 * for apps derived from AutoAppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class AutoWorker {
  static_assert(std::is_base_of<AutoAppBase<typename APP_T::fragment_t,
                                            typename APP_T::context_t>,
                                APP_T>::value,
                "AutoWorker should work with AutoApp");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;

  using message_manager_t =
      AutoParallelMessageManager<typename APP_T::fragment_t>;

  static_assert(check_app_fragment_consistency<APP_T, fragment_t>(),
                "The loaded graph is not valid for application");

  AutoWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(app), context_(std::make_shared<context_t>(*graph)) {}

  ~AutoWorker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    auto& graph = const_cast<fragment_t&>(context_->fragment());
    // prepare for the query
    graph.PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;
    MPI_Barrier(comm_spec_.comm());

    messages_.Init(comm_spec_.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec_.comm());
  }

  void Finalize() {}

  template <class... Args>
  void Query(Args&&... args) {
    auto& graph = context_->fragment();

    MPI_Barrier(comm_spec_.comm());

    context_->Init(messages_, std::forward<Args>(args)...);

    int round = 0;

    messages_.Start();

    messages_.StartARound();

    app_->PEval(graph, *context_);

    messages_.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;

    while (!messages_.ToTerminate()) {
      round++;
      messages_.StartARound();

      app_->IncEval(graph, *context_);

      messages_.FinishARound();

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step;
      }
      ++step;
    }
    MPI_Barrier(comm_spec_.comm());

    messages_.Finalize();
  }

  std::shared_ptr<context_t> GetContext() { return context_; }

  const TerminateInfo& GetTerminateInfo() const {
    return messages_.GetTerminateInfo();
  }

  void Output(std::ostream& os) { context_->Output(os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  message_manager_t messages_;

  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_AUTO_WORKER_H_
