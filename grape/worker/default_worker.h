/* Copyright 2019 Alibaba Group Holding Limited. */

#ifndef GRAPE_WORKER_DEFAULT_WORKER_H_
#define GRAPE_WORKER_DEFAULT_WORKER_H_

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
class DefaultWorker {
  static_assert(std::is_base_of<AppBase<typename APP_T::fragment_t,
                                        typename APP_T::context_t>,
                                APP_T>::value,
                "DefaultWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using context_t = typename APP_T::context_t;

  using message_manager_t = DefaultMessageManager;

  DefaultWorker(std::shared_ptr<APP_T> app, std::shared_ptr<fragment_t> graph)
      : app_(app), graph_(graph) {}

  virtual ~DefaultWorker() {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    // verify the consistency between app and graph
    // prepare for the query
    // 建立一些发消息需要用到的索引，不必深究
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());

    InitParallelEngine(app_, pe_spec);
    InitCommunicator(app_, comm_spec.comm());
  }

  void Finalize() {}

  template <class... Args>
  void Query(Args&&... args) {
    MPI_Barrier(comm_spec_.comm());

    context_ = std::make_shared<context_t>();
    // 调用app的Init方法，初始化app需要用到的数据，例如SSSPSerialContext.Init
    context_->Init(*graph_, messages_, std::forward<Args>(args)...);

    int round = 0;

    messages_.Start();

    // 等待上一轮消息发送/接受完毕，清理一些member
    messages_.StartARound();

    // 调用App的PEval
    app_->PEval(*graph_, *context_, messages_);

    // 接受其他worker发送的消息；发送消息给其他的worker（异步）
    messages_.FinishARound();

    if (comm_spec_.worker_id() == kCoordinatorRank) {
      VLOG(1) << "[Coordinator]: Finished PEval";
    }

    int step = 1;

    // 不断执行IncEval直到收敛
    while (!messages_.ToTerminate()) {
      round++;
      messages_.StartARound();

      app_->IncEval(*graph_, *context_, messages_);

      messages_.FinishARound();

      if (comm_spec_.worker_id() == kCoordinatorRank) {
        VLOG(1) << "[Coordinator]: Finished IncEval - " << step;
      }
      ++step;
    }
    LOG(INFO) << "step: " << step;
    MPI_Barrier(comm_spec_.comm());

    messages_.Finalize();
  }

  void Output(std::ostream& os) { context_->Output(*graph_, os); }

 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> graph_;
  std::shared_ptr<context_t> context_;
  message_manager_t messages_;

  CommSpec comm_spec_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_DEFAULT_WORKER_H_
