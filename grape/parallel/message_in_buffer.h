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

#ifndef GRAPE_PARALLEL_MESSAGE_IN_BUFFER_H_
#define GRAPE_PARALLEL_MESSAGE_IN_BUFFER_H_

#include "grape/serialization/out_archive.h"

namespace grape {
/**
 * @brief MessageInBuffer holds a grape::OutArchive, which contains a bunch of
 * messages. Used By JavaParallelMessageManager to process messages in a
 * parallel manner.
 *
 */
class MessageInBuffer {
 public:
  MessageInBuffer() {}

  explicit MessageInBuffer(OutArchive&& arc) : arc_(std::move(arc)) {}

  void Init(OutArchive&& arc) { arc_ = std::move(arc); }

  template <typename MESSAGE_T>
  inline bool GetMessage(MESSAGE_T& msg) {
    if (arc_.Empty()) {
      return false;
    }
    arc_ >> msg;
    return true;
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  inline bool GetMessage(const GRAPH_T& frag, typename GRAPH_T::vertex_t& v,
                         MESSAGE_T& msg) {
    if (arc_.Empty()) {
      return false;
    }
    typename GRAPH_T::vid_t gid;
    arc_ >> gid >> msg;
    frag.Gid2Vertex(gid, v);
    return true;
  }

 private:
  OutArchive arc_;
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_MESSAGE_IN_BUFFER_H_
