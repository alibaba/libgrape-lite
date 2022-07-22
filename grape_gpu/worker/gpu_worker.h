#ifndef GRAPE_GPU_WORKER_GPU_WORKER_H_
#define GRAPE_GPU_WORKER_GPU_WORKER_H_
#include <functional>
#include <utility>

#include "examples/timer.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/util.h"
#include "grape_gpu/communication/communicator.h"
#include "grape_gpu/parallel/gpu_message_manager.h"

namespace grape_gpu {

template <typename APP_T>
class GPUWorker {
 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;
  using message_manager_t = GPUMessageManager;

  static_assert(grape::check_app_fragment_consistency<APP_T, fragment_t>(),
                "The loaded graph is not valid for application");

  GPUWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(std::move(app)),
        context_(std::make_shared<context_t>(*graph)),
        messages_() {}

  ~GPUWorker() { messages_.Finalize(); }

  template <class... Args>
  void Init(const grape::CommSpec& comm_spec, Args&&... args) {
    auto& graph = const_cast<fragment_t&>(context_->fragment());
    // prepare for the query
    graph.PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges,
                          APP_T::need_build_device_vm);

    comm_spec_ = comm_spec;

    messages_.Init(comm_spec);

    InitCommunicator(app_, comm_spec.comm(), messages_.nccl_comm());

    context_->Init(messages_, std::forward<Args>(args)...);
    app_->Init(graph);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished Init";
    }
  }

  void Finalize() {}

  void Query() {
    double begin;
    auto& graph = context_->fragment();

    messages_.Start();

    begin = grape::GetCurrentTime();
    messages_.StartARound();
    messages_.RecordStarARoundTime(grape::GetCurrentTime() - begin);

    app_->PEval(graph, *context_, messages_);

    messages_.FinishARound();

    begin = grape::GetCurrentTime();
    MPI_Barrier(comm_spec_.comm());
    messages_.RecordBarrierTime(grape::GetCurrentTime() - begin);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;

    auto eval = [&](const std::string& stage,
                    std::function<void(const fragment_t&, context_t&,
                                       message_manager_t&)>
                        f) {
      auto iter_begin = grape::GetCurrentTime();
      begin = grape::GetCurrentTime();
      messages_.StartARound();
      messages_.RecordStarARoundTime(grape::GetCurrentTime() - begin);

      f(graph, *context_, messages_);

      messages_.FinishARound();

      begin = grape::GetCurrentTime();
      MPI_Barrier(comm_spec_.comm());
      messages_.RecordBarrierTime(grape::GetCurrentTime() - begin);

      if (graph.fid() == 0) {
        VLOG(1) << "[Coordinator]: Finished " << stage << " - " << step
                << " Time: " << (grape::GetCurrentTime() - iter_begin) * 1000
                << " ms";
      }
      ++step;
    };

    while (!messages_.ToTerminate()) {
      eval("IncEval", std::bind(&APP_T::IncEval, *app_, std::placeholders::_1,
                                std::placeholders::_2, std::placeholders::_3));
    }

    eval("MigrateSend",
         std::bind(&APP_T::MigrateSend, *app_, std::placeholders::_1,
                   std::placeholders::_2, std::placeholders::_3));
    eval("MigrateRecv",
         std::bind(&APP_T::MigrateRecv, *app_, std::placeholders::_1,
                   std::placeholders::_2, std::placeholders::_3));
    eval("FuseEval", std::bind(&APP_T::FuseEval, *app_, std::placeholders::_1,
                               std::placeholders::_2, std::placeholders::_3));

    Finalize();
  }

  std::shared_ptr<context_t> GetContext() { return context_; }

  void Output(std::ostream& os) { context_->Output(os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  message_manager_t messages_;
  grape::CommSpec comm_spec_;
};
}  // namespace grape_gpu
#endif  // GRAPE_GPU_WORKER_GPU_WORKER_H_
