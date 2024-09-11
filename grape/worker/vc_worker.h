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

#ifndef GRAPE_VC_WORKER_WORKER_H_
#define GRAPE_VC_WORKER_WORKER_H_

#include "grape/app/vc_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/worker/comm_spec.h"

namespace grape {

template <typename APP_T, typename MESSAGE_MANAGER_T>
class VCWorker {
 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;

  using message_manager_t = MESSAGE_MANAGER_T;

  VCWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(app), context_(nullptr), fragment_(graph) {}
  ~VCWorker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    auto& graph = *fragment_;

    comm_spec_ = comm_spec;
    MPI_Barrier(comm_spec_.comm());
    if (context_ == nullptr) {
      context_ = std::make_shared<context_t>(graph);
    }

    messages_.Init(comm_spec_.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec_.comm());
  }

  void Finalize() {}

  template <class... Args>
  void Query(Args&&... args) {
    double t = GetCurrentTime();

    context_->Init(messages_, std::forward<Args>(args)...);
    auto& graph = context_->fragment();

    messages_.Start();

    messages_.StartARound();

    app_->PEval(graph, *context_, messages_);

    messages_.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval, time: " << GetCurrentTime() - t
              << " sec";
    }

    int step = 1;

    while (!messages_.ToTerminate()) {
      t = GetCurrentTime();
      messages_.StartARound();

      app_->IncEval(graph, *context_, messages_);

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
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  std::shared_ptr<fragment_t> fragment_;
  message_manager_t messages_;

  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_VC_WORKER_WORKER_H_
