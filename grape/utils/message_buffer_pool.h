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

#ifndef GRAPE_UTILS_MESSAGE_BUFFER_POOL_H_
#define GRAPE_UTILS_MESSAGE_BUFFER_POOL_H_

#include <deque>

#include "grape/config.h"
#include "grape/utils/concurrent_queue.h"

namespace grape {

struct MicroBuffer {
  MicroBuffer() : buffer(NULL), size(0) {}
  MicroBuffer(char* buffer, size_t size) : buffer(buffer), size(size) {}
  ~MicroBuffer() = default;

  char* buffer;
  size_t size;
};

struct MessageBuffer : public Allocator<char> {
  MessageBuffer() : buffer(NULL), size(0) {}
  MessageBuffer(MessageBuffer&& rhs) : buffer(rhs.buffer), size(rhs.size) {
    rhs.buffer = NULL;
    rhs.size = 0;
  }
  ~MessageBuffer() {
    if (buffer) {
      this->deallocate(buffer, size);
    }
  }

  MessageBuffer& operator=(MessageBuffer&& rhs) {
    if (this != &rhs) {
      if (buffer) {
        this->deallocate(buffer, size);
      }
      buffer = rhs.buffer;
      size = rhs.size;
      rhs.buffer = NULL;
      rhs.size = 0;
    }
    return *this;
  }

  void init(size_t sz) {
    if (buffer) {
      this->deallocate(buffer, size);
    }
    buffer = this->allocate(sz);
    size = sz;
  }

  void reset() {
    if (buffer) {
      this->deallocate(buffer, size);
    }
    buffer = NULL;
    size = 0;
  }

  void set0() {
    if (buffer) {
      memset(buffer, 0, size);
    }
  }

  void swap(MessageBuffer& rhs) {
    std::swap(buffer, rhs.buffer);
    std::swap(size, rhs.size);
  }

  char* buffer;
  size_t size;
};

static constexpr size_t kDefaultPoolBatchSize = 16 * 1024 * 1024;

inline size_t estimate_pool_size(size_t send_message_size,
                                 size_t recv_message_size, size_t batch_size,
                                 size_t fnum, size_t thread_num) {
  size_t channel_num = fnum * thread_num;
  size_t send_pool_size = (send_message_size + (batch_size - 1) * channel_num) /
                          batch_size * batch_size;
  send_pool_size += (send_pool_size + fnum - 1) / fnum;
  send_pool_size =
      std::max(send_pool_size, batch_size * (fnum + 1) * thread_num);

  size_t recv_pool_size =
      (recv_message_size + batch_size - 1) / batch_size * batch_size;
  return send_pool_size + recv_pool_size;
}

class MessageBufferPool {
 public:
  MessageBufferPool()
      : init_size_(0),
        chunk_size_(2ull * 1024 * 1024),
        used_size_(0),
        peak_used_size_(0),
        extra_used_size_(0),
        peak_extra_used_size_(0) {}
  ~MessageBufferPool() {}

  void init(size_t size, size_t chunk) {
    size = std::min(size, static_cast<size_t>(get_available_memory() * 0.6));
    size_t num = size / chunk;
    lock_.lock();
    que_.clear();
    init_size_ = size;
    chunk_size_ = chunk;

    for (size_t i = 0; i < num; ++i) {
      MessageBuffer buf;
      buf.init(chunk);
      buf.set0();
      que_.emplace_back(std::move(buf));
    }
    lock_.unlock();

    used_size_ = 0;
    peak_used_size_ = 0;
    extra_used_size_ = 0;
    peak_extra_used_size_ = 0;
  }

  MessageBuffer take(size_t expect_chunk) {
    if (expect_chunk > chunk_size_) {
      MessageBuffer buf;
      buf.init(expect_chunk);
      lock_.lock();
      extra_used_size_ += expect_chunk;
      peak_extra_used_size_ = std::max(peak_extra_used_size_, extra_used_size_);
      lock_.unlock();
      return buf;
    }
    lock_.lock();
    used_size_ += chunk_size_;
    peak_used_size_ = std::max(peak_used_size_, used_size_);
    if (que_.empty()) {
      lock_.unlock();
      MessageBuffer ret;
      ret.init(chunk_size_);
      return ret;
    } else {
      MessageBuffer ret = std::move(que_.front());
      que_.pop_front();
      lock_.unlock();
      return ret;
    }
  }

  MessageBuffer take_default() { return take(chunk_size_); }

  void give(MessageBuffer&& buf) {
    size_t buf_size = buf.size;
    if (buf_size == chunk_size_) {
      lock_.lock();
      used_size_ -= chunk_size_;
      que_.emplace_back(std::move(buf));
      lock_.unlock();
    } else {
      buf.reset();
      lock_.lock();
      extra_used_size_ -= buf_size;
      lock_.unlock();
    }
  }

  size_t chunk_size() const { return chunk_size_; }

 private:
  SpinLock lock_;
  std::deque<MessageBuffer> que_;

  size_t init_size_;
  size_t chunk_size_;

  size_t used_size_;
  size_t peak_used_size_;

  size_t extra_used_size_;
  size_t peak_extra_used_size_;
};

}  // namespace grape

#endif  // GRAPE_UTILS_MESSAGE_BUFFER_POOL_H_
