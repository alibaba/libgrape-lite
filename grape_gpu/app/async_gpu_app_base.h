#ifndef GRAPE_GPU_APP_GPU_APP_BASE_H_
#define GRAPE_GPU_APP_GPU_APP_BASE_H_

#include <memory>

#include "grape/types.h"
#include "grape_gpu/parallel/async_gpu_message_manager.h"
#include "grape_gpu/worker/async_gpu_worker.h"

namespace grape_gpu {
//template<typename T>
//class AsyncGPUMessageManager;
//
//template<typename T>
//class AsyncGPUWorker;

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
class AsyncGPUAppBase {
 public:
  static constexpr bool need_split_edges = false;
  static constexpr bool need_build_device_vm = false;
  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kSyncOnOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;

  typedef typename CONTEXT_T::msg_t MSG_T;
  using message_manager_t = AsyncGPUMessageManager<MSG_T>;

  AsyncGPUAppBase() = default;
  virtual ~AsyncGPUAppBase() = default;

  virtual void PEval(const FRAG_T& graph, CONTEXT_T& context,
                     message_manager_t& messages) = 0;

  virtual void Unpack(const FRAG_T& graph, CONTEXT_T& context,
                      message_manager_t& messages) = 0;

  virtual void Compute(const FRAG_T& graph, CONTEXT_T& context,
                       message_manager_t& messages) = 0;
};

#define INSTALL_ASYNC_GPU_WORKER(APP_T, CONTEXT_T, FRAG_T)        \
 public:                                                          \
  using fragment_t = FRAG_T;                                      \
  using context_t = CONTEXT_T;                                    \
  using worker_t = grape_gpu::AsyncGPUWorker<APP_T>;              \
  using msg_t = typename CONTEXT_T::msg_t;                         \
  using message_manager_t = grape_gpu::AsyncGPUMessageManager<msg_t>;         \
  using dev_message_manager_t = grape_gpu::dev::AsyncMessageManager<msg_t>;   \
  virtual ~APP_T() {}                                             \
  static std::shared_ptr<worker_t> CreateWorker(                  \
      std::shared_ptr<APP_T> app, std::shared_ptr<FRAG_T> frag) { \
    return std::shared_ptr<worker_t>(new worker_t(app, frag));    \
  }
}  // namespace grape_gpu
#endif  // GRAPE_GPU_APP_ASYNC_GPU_APP_BASE_H_
