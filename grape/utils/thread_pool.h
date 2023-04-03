/** Credits to Jakob Progsch (@progschj)
 https://github.com/progschj/ThreadPool

 Copyright (c) 2012 Jakob Progsch, Václav Zeman

 This software is provided 'as-is', without any express or implied
 warranty. In no event will the authors be held liable for any damages
 arising from the use of this software.

 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:

 1. The origin of this software must not be misrepresented; you must not
 claim that you wrote the original software. If you use this software
 in a product, an acknowledgment in the product documentation would be
 appreciated but is not required.

 2. Altered source versions must be plainly marked as such, and must not be
 misrepresented as being the original software.

 3. This notice may not be removed or altered from any source
 distribution.

 Modified by Binrui Li at Alibaba Group, 2021.
*/

#ifndef GRAPE_UTILS_THREAD_POOL_H_
#define GRAPE_UTILS_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include <glog/logging.h>

#include "grape/parallel/parallel_engine_spec.h"

#if __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#endif

class ThreadPool {
 public:
  ThreadPool(ThreadPool const&) = delete;
  ThreadPool& operator=(ThreadPool const&) = delete;
  ThreadPool() {}
  inline void InitThreadPool(const grape::ParallelEngineSpec&);

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  inline int GetThreadNum() { return thread_num_; }
  void WaitEnd(std::vector<std::future<void>>& results);
  ~ThreadPool();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop{false};
  size_t thread_num_{1};
};

// the constructor just launches some amount of workers
inline void ThreadPool::InitThreadPool(const grape::ParallelEngineSpec& spec) {
  bool affinity = false;
  affinity = spec.affinity && (!spec.cpu_list.empty());
  thread_num_ = spec.thread_num;
  for (size_t i = 0; i < thread_num_; ++i) {
    workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        task();
      }
    });
#if __linux__
    if (affinity) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(spec.cpu_list[i], &cpuset);
      pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t),
                             &cpuset);
      VLOG(2) << "bind thread " << i << " to " << spec.cpu_list[i] << std::endl;
    }
#else
    (void) affinity;
#endif
  }
}

// use to wait for all tasks end
inline void ThreadPool::WaitEnd(std::vector<std::future<void>>& results) {
  for (size_t tid = 0; tid < thread_num_; ++tid)
    results[tid].get();
}

// add new task to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers)
    worker.join();
}

#endif  // GRAPE_UTILS_THREAD_POOL_H_
