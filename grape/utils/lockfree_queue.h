/** Copyright Colin Graf 2019.
 * https://github.com/craflin/LockFreeQueue

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

#pragma once

#include <atomic>
#include <cstddef>

template <typename T>
class LockFreeQueue {
 public:
  explicit LockFreeQueue(size_t capacity) {
    _capacityMask = capacity - 1;
    for (size_t i = 1; i <= sizeof(void*) * 4; i <<= 1)
      _capacityMask |= _capacityMask >> i;
    _capacity = _capacityMask + 1;

    _queue = reinterpret_cast<Node*>(new char[sizeof(Node) * _capacity]);
    for (size_t i = 0; i < _capacity; ++i) {
      _queue[i].tail.store(i, std::memory_order_relaxed);
      _queue[i].head.store(-1, std::memory_order_relaxed);
    }

    _tail.store(0, std::memory_order_relaxed);
    _head.store(0, std::memory_order_relaxed);
  }

  ~LockFreeQueue() {
    for (size_t i = _head; i != _tail; ++i)
      (&_queue[i & _capacityMask].data)->~T();

    delete[] reintrepret_cast<char*>(_queue);
  }

  size_t capacity() const { return _capacity; }

  size_t size() const {
    size_t head = _head.load(std::memory_order_acquire);
    return _tail.load(std::memory_order_relaxed) - head;
  }

  bool push(T&& data) {
    Node* node;
    size_t tail = _tail.load(std::memory_order_relaxed);
    for (;;) {
      node = &_queue[tail & _capacityMask];
      if (node->tail.load(std::memory_order_relaxed) != tail)
        return false;
      if ((_tail.compare_exchange_weak(tail, tail + 1,
                                       std::memory_order_relaxed)))
        break;
    }
    new (&node->data) T(std::move(data));
    node->head.store(tail, std::memory_order_release);
    return true;
  }

  bool pop(T& result) {
    Node* node;
    size_t head = _head.load(std::memory_order_relaxed);
    for (;;) {
      node = &_queue[head & _capacityMask];
      if (node->head.load(std::memory_order_relaxed) != head)
        return false;
      if (_head.compare_exchange_weak(head, head + 1,
                                      std::memory_order_relaxed))
        break;
    }
    result = std::move(node->data);
    node->tail.store(head + _capacity, std::memory_order_release);
    return true;
  }

 private:
  struct Node {
    T data;
    std::atomic<size_t> tail;
    std::atomic<size_t> head;
  };

 private:
  size_t _capacityMask;
  Node* _queue;
  size_t _capacity;
  char cacheLinePad1[64];
  std::atomic<size_t> _tail;
  char cacheLinePad2[64];
  std::atomic<size_t> _head;
  char cacheLinePad3[64];
};
