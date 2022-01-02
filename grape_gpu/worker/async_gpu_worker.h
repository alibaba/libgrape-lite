#ifndef GRAPE_GPU_WORKER_ASYNC_GPU_WORKER_H_
#define GRAPE_GPU_WORKER_ASYNC_GPU_WORKER_H_
#include <utility>
#include <thread>
#include <condition_variable>

#include "examples/timer.h"
#include "grape/util.h"
#include "grape/parallel/parallel_engine.h"
#include "grape_gpu/communication/communicator.h"
#include "grape_gpu/parallel/async_gpu_message_manager.h"

namespace grape_gpu {

template <typename APP_T>
class AsyncGPUWorker {
 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;
  using message_manager_t = typename APP_T::message_manager_t;
  using msg_t = typename APP_T::msg_t;

  static_assert(grape::check_app_fragment_consistency<APP_T, fragment_t>(),
                "The loaded graph is not valid for application");

  AsyncGPUWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(std::move(app)),
        context_(std::make_shared<context_t>(*graph)),
        messages_() {}

  template <class... Args>
  void Init(const grape::CommSpec& comm_spec, Args&&... args) {
    auto& graph = const_cast<fragment_t&>(context_->fragment());
    // prepare for the query
    graph.PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges,
                          APP_T::need_build_device_vm);

    comm_spec_ = comm_spec;

    messages_.Init(comm_spec);

    //InitCommunicator(app_, comm_spec.comm(), messages_.nccl_comm());

    context_->Init(messages_, std::forward<Args>(args)...);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished Init";
    }

    this->running_ = true;
    this->cv_handler_ = 0;
  }

  void Finalize() {}

  void Run() {
    comm_thread_ = std::thread([this]() { CommLoop(); });
    comp_thread_ = std::thread([this]() { CompLoop(); });
  }

  void CommLoop() {
    auto& graph = context_->fragment();
    while (running_) {
    // TODO: do communication work 
      if(this->comm_spec_.fid() == 0) {
        VLOG(2) << "communication 0";
      }
      messages_.ProcessIncomingMsg(); 

      if(this->comm_spec_.fid() == 0) {
        VLOG(2) << "communication 1";
      }
      app_->Unpack(graph, *context_, messages_); // still comm_stream;
      
      messages_.template ProcessOutGoingMsg<msg_t>();

      if(this->comm_spec_.fid() == 0) {
        VLOG(2) << "communication 2";
      }
      // into vote to terminate process.
      if (!cv_handler_) {
        //std::unique_lock<mutex> lck(mtx_);
        auto done = messages_.VoteToTerminate ();
        if (done) running_ = false;
        cv_handler_ = 0;
        cv_.notify_one();
      }
    }
  }

  void CompLoop(){
    auto& graph = context_->fragment();
    while (running_) {
    // TODO: do computation work
      app_->Compute(graph, *context_, messages_);
      // if local == 0, hang on a cv.
      if (messages_.IsCompEmpty()) {
        std::unique_lock<std::mutex> lck(mtx_);
        cv_handler_ = 1;
        cv_.wait(lck, [=]{return this->cv_handler_ == 0;});
      }
    }
  }

  void Query() {
    auto& graph = context_->fragment();
    //VLOG(2) << "Worker " << comm_spec_.fid() << ": Start Query";
    messages_.Barrier();
    app_->PEval(graph, *context_, messages_);
    messages_.Barrier();
    VLOG(2) << "Worker " << comm_spec_.fid() << ": Start Run";
    Run();

    comp_thread_.join();
    comm_thread_.join();

    messages_.Finalize();
  }

  std::shared_ptr<context_t> GetContext() { return context_; }

  void Output(std::ostream& os) { context_->Output(os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  message_manager_t messages_;
  grape::CommSpec comm_spec_;

  // for async collaboration
  std::thread comp_thread_;
  std::thread comm_thread_;

  std::atomic<bool> running_;
  std::condition_variable cv_;
  std::mutex mtx_;
  size_t cv_handler_;
};
}  // namespace grape_gpu
#endif  // GRAPE_GPU_WORKER_GPU_WORKER_H_
