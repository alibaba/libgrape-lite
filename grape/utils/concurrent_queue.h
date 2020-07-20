/** Copyright 2020 Alibaba Group Holding Limited.

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

#ifndef GRAPE_UTILS_CONCURRENT_QUEUE_H_
#define GRAPE_UTILS_CONCURRENT_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <utility>

namespace grape {

/**
 * @brief A concurrent queue based on condition_variables and can be accessed by
 * multi-producers and multi-consumers simultaneously.
 *
 * @tparam T Type of entities in the queue.
 */
template <typename T>
class BlockingQueue {
 public:
  BlockingQueue() : size_limit_(std::numeric_limits<size_t>::max()) {}
  ~BlockingQueue() {}

  /**
   * @brief Set queue size. When queue_.size() == size_limit_, Putting entities
   * will be blocked.
   *
   * @param limit Size limit of the queue.
   */
  void SetLimit(size_t limit) { size_limit_ = limit; }

  /**
   * @brief When a producer finished producing, it will call this function.
   * When all producers finished producing, blocked consumer will be notified to
   * return.
   */
  void DecProducerNum() {
    {
      std::unique_lock<std::mutex> lk(lock_);
      --producer_num_;
    }
    if (producer_num_ == 0) {
      empty_.notify_all();
    }
  }

  /**
   * @brief Set the number of producers to this queue.
   *
   * This function is supposed to be called before producers start to put
   * entities into this queue.
   *
   * @param pn Number of producers to this queue.
   */
  void SetProducerNum(int pn) { producer_num_ = pn; }

  /**
   * @brief Put an entity into this queue.
   *
   * This function will be blocked when the queue is full, that is,
   * queue_.size() == size_limit_.
   *
   * @param item The entity to be put.
   */
  void Put(const T& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.size() >= size_limit_) {
        full_.wait(lk);
      }
      queue_.emplace_back(item);
    }
    empty_.notify_one();
  }

  /**
   * @brief Put an entity into this queue.
   *
   * This function will be blocked when the queue is full, that is,
   * queue_.size() == size_limit_.
   *
   * @param item The entity to be put.
   */
  void Put(T&& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.size() >= size_limit_) {
        full_.wait(lk);
      }
      queue_.emplace_back(std::move(item));
    }
    empty_.notify_one();
  }

  /**
   * @brief Get an entity from this queue.
   *
   * This function will be blocked when there are alive producers and the queue
   * is empty, and will be waken when entities are put into the queue or all
   * producers finished putting data.
   *
   * @param item Reference of an entity to hold the got data.
   *
   * @return If got data, return true. Otherwise, return false.
   */
  bool Get(T& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.empty() && (producer_num_ != 0)) {
        empty_.wait(lk);
      }
      if (queue_.empty() && (producer_num_ == 0)) {
        return false;
      } else {
        item = std::move(queue_.front());
        queue_.pop_front();
        full_.notify_one();
        return true;
      }
    }
  }

  size_t Size() const { return queue_.size(); }

 private:
  std::deque<T> queue_;
  size_t size_limit_;
  std::mutex lock_;
  std::condition_variable empty_, full_;

  std::atomic<int> producer_num_;
};

/**
 * @brief A simple implementation of spinlock based on std::atomic.
 */
class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

 public:
  void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
      {
      }
    }
  }

  void unlock() { locked.clear(std::memory_order_relaxed); }
};

/**
 * @brief A concurrent queue guarded by a spinlock and can be accessed by
 * multi-producers and multi-consumers simultaneously.
 *
 * @tparam T Type of entities in the queue.
 */
template <typename T>
class NonblockingQueue {
 public:
  NonblockingQueue() {}
  ~NonblockingQueue() {}

  /**
   * @brief Put an entity into this queue.
   *
   * This function won't be blocked.
   *
   * @param item The entity to be put.
   */
  void Put(const T& item) {
    lock_.lock();
    queue_.emplace_back(item);
    lock_.unlock();
  }

  /**
   * @brief Put an entity into this queue.
   *
   * This function won't be blocked.
   *
   * @param item The entity to be put.
   */
  void Put(T&& item) {
    lock_.lock();
    queue_.emplace_back(std::move(item));
    lock_.unlock();
  }

  /**
   * @brief Get an entity from this queue.
   *
   * This function won't be blocked.
   *
   * @param item Reference of an entity to hold the got data.
   *
   * @return If got data, return true. Otherwise, return false.
   */
  bool Get(T& item) {
    bool ret = false;
    lock_.lock();
    if (!queue_.empty()) {
      ret = true;
      item = std::move(queue_.front());
      queue_.pop_front();
    }
    lock_.unlock();
    return ret;
  }

  void Clear() {
    lock_.lock();
    queue_.clear();
    lock_.unlock();
  }

 private:
  std::deque<T> queue_;
  SpinLock lock_;
};

}  // namespace grape

#endif  // GRAPE_UTILS_CONCURRENT_QUEUE_H_
