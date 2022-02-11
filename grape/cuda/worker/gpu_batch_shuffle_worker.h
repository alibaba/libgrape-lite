/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef GRAPE_CUDA_WORKER_GPU_BATCH_SHUFFLE_WORKER_H_
#define GRAPE_CUDA_WORKER_GPU_BATCH_SHUFFLE_WORKER_H_
#include <utility>

#include "grape/cuda/communication/communicator.h"
#include "grape/cuda/parallel/batch_shuffle_message_manager.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/util.h"
#include "grape/worker/worker.h"

namespace grape {
namespace cuda {
template <typename APP_T>
class GPUBatchShuffleWorker {
 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;
  using message_manager_t = BatchShuffleMessageManager;

  static_assert(grape::check_app_fragment_consistency<APP_T, fragment_t>(),
                "The loaded graph is not valid for application");

  GPUBatchShuffleWorker(std::shared_ptr<APP_T> app,
                        std::shared_ptr<fragment_t> graph)
      : app_(std::move(app)),
        context_(std::make_shared<context_t>(*graph)),
        messages_() {}

  template <class... Args>
  void Init(const grape::CommSpec& comm_spec, Args&&... args) {
    auto& graph = const_cast<fragment_t&>(context_->fragment());
    // prepare for the query
    PrepareConf prepare_conf;
    prepare_conf.message_strategy = APP_T::message_strategy;
    prepare_conf.need_split_edges = APP_T::need_split_edges;
    prepare_conf.need_mirror_info = true;
    prepare_conf.need_build_device_vm = APP_T::need_build_device_vm;
    graph.PrepareToRunApp(comm_spec, prepare_conf);

    comm_spec_ = comm_spec;

    messages_.Init(comm_spec);

    InitCommunicator(app_, comm_spec.comm(), messages_.nccl_comm());

    context_->Init(messages_, std::forward<Args>(args)...);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished Init";
    }
  }

  void Finalize() {}

  void Query() {
    auto& graph = context_->fragment();

    messages_.Start();

    messages_.StartARound();

    app_->PEval(graph, *context_, messages_);

    messages_.FinishARound();

    MPI_Barrier(comm_spec_.comm());

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;

    while (!messages_.ToTerminate()) {
      auto iter_begin = grape::GetCurrentTime();
      messages_.StartARound();

      app_->IncEval(graph, *context_, messages_);

      messages_.FinishARound();

      MPI_Barrier(comm_spec_.comm());

      if (graph.fid() == 0) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step
                << " Time: " << (grape::GetCurrentTime() - iter_begin) * 1000
                << " ms";
      }
      ++step;
    }

    messages_.Finalize();
  }

  std::shared_ptr<context_t> GetContext() { return context_; }

  void Output(std::ostream& os) { context_->Output(os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  message_manager_t messages_;
  grape::CommSpec comm_spec_;
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_WORKER_GPU_BATCH_SHUFFLE_WORKER_H_
