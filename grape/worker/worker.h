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

#ifndef GRAPE_WORKER_WORKER_H_
#define GRAPE_WORKER_WORKER_H_

#include <mpi.h>

#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>

#include "grape/app/mutation_context.h"
#include "grape/communication/communicator.h"
#include "grape/config.h"
#include "grape/parallel/auto_parallel_message_manager.h"
#include "grape/parallel/batch_shuffle_message_manager.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/parallel/parallel_message_manager.h"
#include "grape/parallel/parallel_message_manager_opt.h"
#include "grape/util.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief A Worker manages the computation cycle.
 *
 * @tparam APP_T
 * @tparam MESSAGE_MANAGER_T
 */
template <typename APP_T, typename MESSAGE_MANAGER_T>
class Worker {
 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;

  using message_manager_t = MESSAGE_MANAGER_T;

  static_assert(check_app_fragment_consistency<APP_T, fragment_t>(),
                "The loaded graph is not valid for application");

  Worker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(app),
        context_(std::make_shared<context_t>(*graph)),
        fragment_(graph) {
    prepare_conf_.message_strategy = APP_T::message_strategy;
    prepare_conf_.need_split_edges = APP_T::need_split_edges;
    prepare_conf_.need_split_edges_by_fragment =
        APP_T::need_split_edges_by_fragment;
    prepare_conf_.need_mirror_info =
        std::is_same<message_manager_t, BatchShuffleMessageManager>::value;
  }

  ~Worker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    auto& graph = const_cast<fragment_t&>(context_->fragment());
    // prepare for the query
    graph.PrepareToRunApp(comm_spec, prepare_conf_);

    comm_spec_ = comm_spec;
    MPI_Barrier(comm_spec_.comm());

    messages_.Init(comm_spec_.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec_.comm());
  }

  void Finalize() {}

  template <class... Args>
  void Query(Args&&... args) {
    double t = GetCurrentTime();

    MPI_Barrier(comm_spec_.comm());

    context_->Init(messages_, std::forward<Args>(args)...);
    processMutation();

    int round = 0;

    messages_.Start();

    messages_.StartARound();

    runPEval();
    processMutation();

    messages_.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval, time: " << GetCurrentTime() - t
              << " sec";
    }

    int step = 1;

    while (!messages_.ToTerminate()) {
      t = GetCurrentTime();
      round++;
      messages_.StartARound();

      runIncEval();
      processMutation();

      messages_.FinishARound();

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step
                << ", time: " << GetCurrentTime() - t << " sec";
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
  template <typename T = message_manager_t>
  typename std::enable_if<
      std::is_same<T, AutoParallelMessageManager<fragment_t>>::value>::type
  runPEval() {
    auto& graph = context_->fragment();
    app_->PEval(graph, *context_);
  }

  template <typename T = message_manager_t>
  typename std::enable_if<
      !std::is_same<T, AutoParallelMessageManager<fragment_t>>::value>::type
  runPEval() {
    auto& graph = context_->fragment();
    app_->PEval(graph, *context_, messages_);
  }

  template <typename T = message_manager_t>
  typename std::enable_if<
      std::is_same<T, AutoParallelMessageManager<fragment_t>>::value>::type
  runIncEval() {
    auto& graph = context_->fragment();
    app_->IncEval(graph, *context_);
  }

  template <typename T = message_manager_t>
  typename std::enable_if<
      !std::is_same<T, AutoParallelMessageManager<fragment_t>>::value>::type
  runIncEval() {
    auto& graph = context_->fragment();
    app_->IncEval(graph, *context_, messages_);
  }

  template <typename T = context_t>
  typename std::enable_if<
      std::is_base_of<MutationContext<fragment_t>, T>::value>::type
  processMutation() {
    context_->apply_mutation(fragment_, comm_spec_);
    fragment_->PrepareToRunApp(comm_spec_, prepare_conf_);
  }

  template <typename T = context_t>
  typename std::enable_if<
      !std::is_base_of<MutationContext<fragment_t>, T>::value>::type
  processMutation() {}

  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  std::shared_ptr<fragment_t> fragment_;
  message_manager_t messages_;

  CommSpec comm_spec_;
  PrepareConf prepare_conf_;
};

template <typename APP_T>
using ParallelWorker = Worker<APP_T, ParallelMessageManager>;

template <typename APP_T>
using ParallelWorkerOpt = Worker<APP_T, ParallelMessageManagerOpt>;

template <typename APP_T>
using AutoWorker =
    Worker<APP_T, AutoParallelMessageManager<typename APP_T::fragment_t>>;

template <typename APP_T>
using BatchShuffleWorker = Worker<APP_T, BatchShuffleMessageManager>;

}  // namespace grape

#endif  // GRAPE_WORKER_WORKER_H_
