/* Copyright 2019 Alibaba Group Holding Limited. */

#ifndef GRAPE_WORKER_TORNADO_WORKER_H_
#define GRAPE_WORKER_TORNADO_WORKER_H_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/default_message_manager.h"
#include "grape/parallel/parallel_engine.h"

namespace grape {

template <typename FRAG_T, typename CONTEXT_T>
class AppBase;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class TornadoWorker {
  static_assert(std::is_base_of<ParallelAppBase<typename APP_T::fragment_t,
                                                typename APP_T::context_t>,
                                APP_T>::value,
                "TornadoWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;

  using message_manager_t = ParallelMessageManager;

  explicit TornadoWorker(std::shared_ptr<APP_T> app)
      : app_(app),
        context_(std::make_shared<context_t>()),
        initialized_(false) {}

  virtual ~TornadoWorker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    comm_spec_ = comm_spec;

    MPI_Barrier(comm_spec_.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec.comm());
  }

  void Finalize() { context_->fragment().reset(); }

  void SetFragment(const std::shared_ptr<fragment_t>& fragment) {
    context_->set_fragment(fragment);
    fragment->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);
  }

  template <class... Args>
  void Query(Args&&... args) {
    auto graph = context_->fragment();
    MPI_Barrier(comm_spec_.comm());

    message_manager_t messages_;
    messages_.Init(comm_spec_.comm());
    if (!initialized_) {
      context_->Init(*graph, messages_, std::forward<Args>(args)...);
      initialized_ = true;
    }

    messages_.Start();

    messages_.StartARound();

    app_->PEval(*graph, *context_, messages_);

    messages_.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;

    while (!messages_.ToTerminate()) {
      messages_.StartARound();

      app_->IncEval(*graph, *context_, messages_);

      messages_.FinishARound();

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step;
      }
      ++step;
    }
    MPI_Barrier(comm_spec_.comm());

    messages_.Finalize();
    MPI_Barrier(comm_spec_.comm());
  }

  void Output(std::ostream& os) { context_->Output(*context_->fragment(), os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  bool initialized_;

  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_TORNADO_WORKER_H_
