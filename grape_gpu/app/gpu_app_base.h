#ifndef GRAPE_GPU_APP_ASYNC_GPU_APP_BASE_H_
#define GRAPE_GPU_APP_ASYNC_GPU_APP_BASE_H_

#include <memory>

#include "grape/types.h"

namespace grape_gpu {
class GPUMessageManager;

template <typename T>
class GPUWorker;

/**
 * @brief GPUAppBase is a base class for GPU apps. Users can process
 * messages in a more flexible way in this kind of app. It contains an
 * GPUMessageManager to process messages, which enables send/receive
 * messages during computation. This strategy improves performance by
 * overlapping the communication time and the evaluation time.
 *
 * @tparam FRAG_T
 * @tparam CONTEXT_T
 */
template <typename FRAG_T, typename CONTEXT_T>
class GPUAppBase {
 public:
  static constexpr bool need_split_edges = false;
  static constexpr bool need_build_device_vm = false;
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kSyncOnOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  using message_manager_t = GPUMessageManager;

  GPUAppBase() = default;
  virtual ~GPUAppBase() = default;

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
  virtual void PEval(const FRAG_T& graph, CONTEXT_T& context,
                     message_manager_t& messages) = 0;

  /**
   * @brief Incremental evaluation to implement.
   *
   * @note: This pure virtual function works as an interface, instructing users
   * to implement in the specific app. The IncEval in the inherited apps would
   * be invoked directly, not via virtual functions.
   *
   * @param graph
   * @param context
   * @param messages
   */
  virtual void IncEval(const FRAG_T& graph, CONTEXT_T& context,
                       message_manager_t& messages) = 0;

  virtual void MigrateSend(const FRAG_T& graph, CONTEXT_T& context,
                           message_manager_t& messages) {}

  virtual void MigrateRecv(const FRAG_T& graph, CONTEXT_T& context,
                           message_manager_t& messages) {}

  virtual void FuseEval(const FRAG_T& graph, CONTEXT_T& context,
                        message_manager_t& messages) {}
};
#define INSTALL_GPU_WORKER(APP_T, CONTEXT_T, FRAG_T)              \
 public:                                                          \
  using fragment_t = FRAG_T;                                      \
  using context_t = CONTEXT_T;                                    \
  using worker_t = grape_gpu::GPUWorker<APP_T>;                   \
  using message_manager_t = grape_gpu::GPUMessageManager;         \
  using dev_message_manager_t = grape_gpu::dev::MessageManager;   \
  virtual ~APP_T() {}                                             \
  static std::shared_ptr<worker_t> CreateWorker(                  \
      std::shared_ptr<APP_T> app, std::shared_ptr<FRAG_T> frag) { \
    return std::shared_ptr<worker_t>(new worker_t(app, frag));    \
  }
}  // namespace grape_gpu
#endif  // GRAPE_GPU_APP_GPU_APP_BASE_H_
