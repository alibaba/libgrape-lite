
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_BLOCKING_QUEUE_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_BLOCKING_QUEUE_H_
#include <condition_variable>
#include <mutex>
#include <queue>

namespace grape_gpu {
template <typename T>
class BlockingQueue {
 private:
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::deque<T> d_queue;

 public:
  void push(T const& value) {
    {
      std::unique_lock<std::mutex> lock(this->d_mutex);
      d_queue.push_front(value);
    }
    this->d_condition.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    this->d_condition.wait(lock, [=] { return !this->d_queue.empty(); });
    T rc(std::move(this->d_queue.back()));
    this->d_queue.pop_back();
    return rc;
  }

  bool empty() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    return d_queue.empty();
  }
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_BLOCKING_QUEUE_H_
