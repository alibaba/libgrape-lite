/* Copyright 2019 Alibaba Group Holding Limited. */

#ifndef GRAPE_WORKER_INC_WORKER_H_
#define GRAPE_WORKER_INC_WORKER_H_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/parallel_message_manager.h"
#include "grape/parallel/parallel_engine.h"

namespace grape {

template <typename FRAG_T, typename CONTEXT_T>
class IncAppBase;

/**
 * @brief A Worker manages the computation cycle. IncDefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class IncWorker {
  static_assert(std::is_base_of<IncAppBase<typename APP_T::fragment_t,
                                           typename APP_T::context_t>,
                                APP_T>::value,
                "IncDefaultWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;
  using oid_t = typename fragment_t::oid_t;
  using message_manager_t = ParallelMessageManager;

  explicit IncWorker(std::shared_ptr<APP_T> app) : app_(app) {}

  virtual ~IncWorker() = default;

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec.comm());
  }

  void Finalize() {  }

  void SetFragment(std::shared_ptr<fragment_t>& graph) {
    graph_ = graph;
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);
  }

  template <class... Args>
  void Query(Args&&... args) {
    MPI_Barrier(comm_spec_.comm());

    context_ = std::make_shared<context_t>();
    message_manager_t messages;

    messages.Init(comm_spec_.comm());

    context_->Init(*graph_, messages, std::forward<Args>(args)...);


    int round = 0;

    messages.Start();

    // 等待上一轮消息发送/接受完毕，清理一些member
    messages.StartARound();

    // 调用App的PEval
    app_->PEval(*graph_, *context_, messages);

    // 接受其他worker发送的消息；发送消息给其他的worker（异步）
    messages.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;
    timer_next("Batch time");
    // 不断执行IncEval直到收敛
    while (!messages.ToTerminate()) {
      round++;
      messages.StartARound();

      app_->IncEval(*graph_, *context_, messages);

      messages.FinishARound();

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step;
      }
      ++step;
    }
    LOG(INFO) << "step: " << step;
    MPI_Barrier(comm_spec_.comm());
    messages.Finalize();
  }

  void Inc(const std::vector<std::pair<oid_t, oid_t>>& added_edges,
           const std::vector<std::pair<oid_t, oid_t>>& deleted_edges) {
    int round = 0;
    message_manager_t messages;

    messages.Init(comm_spec_.comm());

    messages.Start();

    messages.StartARound();

    app_->AdjustPEval(*graph_, added_edges, deleted_edges, *context_,
                      messages);

    messages.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;

    timer_next("Inc time");
    while (!messages.ToTerminate()) {
      round++;
      messages.StartARound();

      app_->AdjustIncEval(*graph_, added_edges, deleted_edges, *context_,
                          messages);

      messages.FinishARound();

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step;
      }
      ++step;
    }
    LOG(INFO) << "step: " << step;
    MPI_Barrier(comm_spec_.comm());
    messages.Finalize();
  }

  void Output(std::ostream& os) { context_->Output(*graph_, os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<context_t> context_;
  std::shared_ptr<fragment_t> graph_;
  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_INC_WORKER_H_
