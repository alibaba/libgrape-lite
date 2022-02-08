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

#ifndef GRAPE_CUDA_UTILS_EVENT_H_
#define GRAPE_CUDA_UTILS_EVENT_H_
#include <cassert>
#include <utility>

#include "grape/cuda/utils/cuda_utils.h"
#include "grape/cuda/utils/stream.h"

namespace grape {
namespace cuda {
struct IEvent {
  virtual ~IEvent() = default;

  virtual void Wait(cudaStream_t stream) const = 0;
  virtual void Sync() const = 0;
  virtual bool Query() const = 0;

  virtual cudaEvent_t cuda_event() const = 0;
};

class EventHolder : public IEvent {
 public:
  EventHolder(cudaEvent_t cuda_event, std::function<void(cudaEvent_t)> releaser)
      : cuda_event_(cuda_event), releaser_(std::move(releaser)) {}

  ~EventHolder() { releaser_(cuda_event_); }

  void Wait(cudaStream_t stream) const override {
    CHECK_CUDA(cudaStreamWaitEvent(stream, cuda_event_, 0));
  }

  void Sync() const override { CHECK_CUDA(cudaEventSynchronize(cuda_event_)); }

  bool Query() const override {
    return cudaEventQuery(cuda_event_) == cudaSuccess;
  }

  cudaEvent_t cuda_event() const { return cuda_event_; }

 private:
  const cudaEvent_t cuda_event_;
  std::function<void(cudaEvent_t)> releaser_;
};

class Event {
 public:
  Event() = default;

  Event(const Event& other) = default;

  Event(Event&& other) noexcept
      : internal_event_(std::move(other.internal_event_)) {}

  Event& operator=(Event&& other) noexcept {
    this->internal_event_ = std::move(other.internal_event_);
    return *this;
  }

  Event& operator=(const Event& other) {
    internal_event_ = other.internal_event_;
    return *this;
  }

  explicit Event(std::shared_ptr<IEvent> internal_event)
      : internal_event_(std::move(internal_event)) {}

  static Event Create() {
    cudaEvent_t cuda_event;
    CHECK_CUDA(cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming));

    return Event(cuda_event,
                 [](cudaEvent_t ev) { CHECK_CUDA(cudaEventDestroy(ev)); });
  }

  void Record(const Stream& stream) const {
    CHECK_CUDA(
        cudaEventRecord(internal_event_->cuda_event(), stream.cuda_stream()));
  }

  void Wait(const Stream& stream) const {
    if (internal_event_ == nullptr)
      return;
    internal_event_->Wait(stream.cuda_stream());
  }

  void Sync() const {
    if (internal_event_ == nullptr)
      return;
    internal_event_->Sync();
  }

  bool Query() const {
    if (internal_event_ == nullptr)
      return true;
    return internal_event_->Query();
  }

 private:
  std::shared_ptr<IEvent> internal_event_{};

  Event(cudaEvent_t cuda_event,
        const std::function<void(cudaEvent_t)>& releaser) {
    assert(cuda_event != nullptr);
    assert(releaser != nullptr);

    internal_event_ = std::make_shared<EventHolder>(cuda_event, releaser);
  }
};
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_EVENT_H_
