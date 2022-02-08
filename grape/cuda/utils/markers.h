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

// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef GRAPE_CUDA_UTILS_MARKERS_H_
#define GRAPE_CUDA_UTILS_MARKERS_H_
namespace grape {
namespace cuda {

#include <nvToolsExt.h>

/**
Example Usage:
{
    RangeMarker marker (true, "Scoped marker");

    {
        RangeMarker another;
        Marker::MarkDouble(1337.5);
        another.Stop();
        Marker::Mark();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
}
*/

static const int CATEGORY_KERNEL_WORKITEMS = 1;
static const int CATEGORY_INTERVAL_WORKITEMS = 2;

struct Marker {
  static nvtxEventAttributes_t CreateEvent(const char* message = "Marker",
                                           int color = 0, int category = 0) {
    nvtxEventAttributes_t eventAttrib = {0};
    // set the version and the size information
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    // configure the attributes.  0 is the default for all attributes.
    if (color) {
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = 0xFF880000;
    }
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    eventAttrib.category = category;

    return eventAttrib;
  }

  static void Mark(const char* message = "Marker", int color = 0,
                   int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkInt(int64_t value, const char* message = "Marker",
                      int color = 0, int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
    ev.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    ev.payload.llValue = value;
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkUnsignedInt(uint64_t value, const char* message = "Marker",
                              int color = 0, int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
    ev.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    ev.payload.ullValue = value;
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkDouble(double value, const char* message = "Marker",
                         int color = 0, int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
    ev.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
    ev.payload.dValue = value;
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkWorkitems(uint64_t items, const char* message) {
    MarkUnsignedInt(items, message, 0, CATEGORY_KERNEL_WORKITEMS);
  }
};

struct RangeMarker {
  bool m_running;
  nvtxEventAttributes_t m_ev;
  nvtxRangeId_t m_id;

  explicit RangeMarker(bool autostart = true, const char* message = "Range",
                       int color = 0, int category = 0)
      : m_running(false), m_id(0) {
    m_ev = Marker::CreateEvent(message, color, category);
    if (autostart)
      Start();
  }

  virtual ~RangeMarker() {
    if (m_running)
      Stop();
  }

  void Start() {
#ifndef DISABLE_NVTX_MARKERS
    if (!m_running) {
      m_id = nvtxRangeStartEx(&m_ev);
      m_running = true;
    }
#endif
  }

  void Stop() {
#ifndef DISABLE_NVTX_MARKERS
    if (m_running) {
      nvtxRangeEnd(m_id);
      m_running = false;
      m_id = 0;
    }
#endif
  }
};

struct IntervalRangeMarker : public RangeMarker {
  explicit IntervalRangeMarker(uint64_t workitems,
                               const char* message = "Range",
                               bool autostart = true)
      : RangeMarker(false, message, 0, CATEGORY_INTERVAL_WORKITEMS) {
    this->m_ev.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    this->m_ev.payload.ullValue = workitems;
    if (autostart)
      Start();
  }

  virtual ~IntervalRangeMarker() {}
};
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_UTILS_MARKERS_H_
