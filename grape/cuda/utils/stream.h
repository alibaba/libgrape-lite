/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef GRAPE_CUDA_UTILS_STREAM_H_
#define GRAPE_CUDA_UTILS_STREAM_H_
#include "grape/cuda/utils/cuda_utils.h"

namespace grape {
namespace cuda {
enum class StreamPriority { kDefault, kHigh, kLow };

class Stream {
 public:
  explicit Stream(StreamPriority priority = StreamPriority::kDefault)
      : priority_(priority) {
    if (priority_ == StreamPriority::kDefault) {
      CHECK_CUDA(
          cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
    } else {
      int leastPriority, greatestPriority;
      CHECK_CUDA(
          cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      CHECK_CUDA(cudaStreamCreateWithPriority(
          &cuda_stream_, cudaStreamNonBlocking,
          priority_ == StreamPriority::kHigh ? greatestPriority
                                             : leastPriority));
    }
  }

  Stream(const Stream& other) = delete;

  Stream(Stream&& other) noexcept
      : priority_(other.priority_), cuda_stream_(other.cuda_stream_) {
    other.cuda_stream_ = nullptr;
  }

  Stream& operator=(const Stream& other) = delete;

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      this->cuda_stream_ = other.cuda_stream_;
      other.cuda_stream_ = nullptr;
    }
    return *this;
  }

  ~Stream() {
    if (cuda_stream_ != nullptr) {
      CHECK_CUDA(cudaStreamDestroy(cuda_stream_));
    }
  }

  void Sync() const { CHECK_CUDA(cudaStreamSynchronize(cuda_stream_)); }

  cudaStream_t cuda_stream() const { return cuda_stream_; }

 private:
  StreamPriority priority_;
  cudaStream_t cuda_stream_{};
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_STREAM_H_
