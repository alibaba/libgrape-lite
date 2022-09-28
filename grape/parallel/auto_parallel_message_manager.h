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

#ifndef GRAPE_PARALLEL_AUTO_PARALLEL_MESSAGE_MANAGER_H_
#define GRAPE_PARALLEL_AUTO_PARALLEL_MESSAGE_MANAGER_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/fragment/edgecut_fragment_base.h"
#include "grape/parallel/default_message_manager.h"
#include "grape/parallel/message_manager_base.h"
#include "grape/parallel/sync_buffer.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/types.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief A kind of message manager supporting auto parallelism.
 *
 * After registering the vertex array and message strategy as a sync buffer,
 * message generation and ingestion can be applied by message manager
 * automatically.
 */
template <typename FRAG_T>
class AutoParallelMessageManager : public DefaultMessageManager {
  using Base = DefaultMessageManager;
  using vid_t = typename FRAG_T::vid_t;

  struct ap_event {
    ap_event(const FRAG_T& f, ISyncBuffer* b, MessageStrategy m, int e)
        : fragment(f), buffer(b), message_strategy(m), event_id(e) {}

    const FRAG_T& fragment;
    ISyncBuffer* buffer;
    MessageStrategy message_strategy;
    int event_id;
  };

 public:
  AutoParallelMessageManager() {}
  ~AutoParallelMessageManager() override {}

  using Base::Init;

  using Base::Start;

  /**
   * @brief Inherit
   */
  void StartARound() override {
    Base::StartARound();
    aggregateAutoMessages();
  }

  /**
   * @brief Inherit
   */
  void FinishARound() override {
    generateAutoMessages();
    Base::FinishARound();
  }

  using Base::ToTerminate;

  using Base::Finalize;

  using Base::GetMsgSize;

  using Base::ForceContinue;

  /**
   * @brief Register a buffer to be sync automatically between rounds.
   *
   * @param frag
   * @param buffer
   * @param strategy
   */
  inline void RegisterSyncBuffer(const FRAG_T& frag, ISyncBuffer* buffer,
                                 MessageStrategy strategy) {
    int event_id = auto_parallel_events_.size();
    auto_parallel_events_.emplace_back(frag, buffer, strategy, event_id);
  }

 private:
  void aggregateAutoMessages() {
    std::map<int, ap_event*> event_map;
    for (auto& event : auto_parallel_events_) {
      event_map.emplace(event.event_id, &event);
    }

    int event_id;
    while (Base::GetMessage<int>(event_id)) {
      ap_event* event = event_map.at(event_id);

      auto& i_ec_frag = event->fragment;
      if (event->message_strategy == MessageStrategy::kSyncOnOuterVertex ||
          event->message_strategy == MessageStrategy::kAlongEdgeToOuterVertex ||
          event->message_strategy ==
              MessageStrategy::kAlongOutgoingEdgeToOuterVertex ||
          event->message_strategy ==
              MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
        if (event->buffer->GetTypeId() == typeid(double)) {
          syncOnVertexRecv<double>(i_ec_frag, event->buffer);
        } else if (event->buffer->GetTypeId() == typeid(uint32_t)) {
          syncOnVertexRecv<uint32_t>(i_ec_frag, event->buffer);
        } else if (event->buffer->GetTypeId() == typeid(int32_t)) {
          syncOnVertexRecv<int32_t>(i_ec_frag, event->buffer);
        } else if (event->buffer->GetTypeId() == typeid(int64_t)) {
          syncOnVertexRecv<int64_t>(i_ec_frag, event->buffer);
        } else if (event->buffer->GetTypeId() == typeid(uint64_t)) {
          syncOnVertexRecv<uint64_t>(i_ec_frag, event->buffer);
        } else if (event->buffer->GetTypeId() ==
                   typeid(std::vector<uint32_t>)) {
          syncOnVertexRecv<std::vector<uint32_t>>(i_ec_frag, event->buffer);
        } else if (event->buffer->GetTypeId() ==
                   typeid(std::vector<uint64_t>)) {
          syncOnVertexRecv<std::vector<uint64_t>>(i_ec_frag, event->buffer);
        } else {
          LOG(FATAL) << "Unexpected data type "
                     << event->buffer->GetTypeId().name();
        }
      } else {
        LOG(FATAL) << "Unexpected message stratety "
                   << underlying_value(event->message_strategy);
      }
    }
  }

  void generateAutoMessages() {
    for (auto& event_ref : auto_parallel_events_) {
      ap_event* event = &event_ref;
      auto& i_ec_frag = event->fragment;
      auto inner_size = i_ec_frag.InnerVertices().size();
      if (event->buffer->updated(0, inner_size)) {
        ForceContinue();
        break;
      }
    }

    for (auto& event_ref : auto_parallel_events_) {
      ap_event* event = &event_ref;

      auto& i_ec_frag = event->fragment;
      if (event->message_strategy == MessageStrategy::kSyncOnOuterVertex) {
        if (event->buffer->GetTypeId() == typeid(double)) {
          syncOnOuterVertexSend<double>(i_ec_frag, event->buffer,
                                        event->event_id);
        } else if (event->buffer->GetTypeId() == typeid(uint32_t)) {
          syncOnOuterVertexSend<uint32_t>(i_ec_frag, event->buffer,
                                          event->event_id);
        } else if (event->buffer->GetTypeId() == typeid(int32_t)) {
          syncOnOuterVertexSend<int32_t>(i_ec_frag, event->buffer,
                                         event->event_id);
        } else if (event->buffer->GetTypeId() == typeid(int64_t)) {
          syncOnOuterVertexSend<int64_t>(i_ec_frag, event->buffer,
                                         event->event_id);
        } else if (event->buffer->GetTypeId() == typeid(uint64_t)) {
          syncOnOuterVertexSend<uint64_t>(i_ec_frag, event->buffer,
                                          event->event_id);
        } else {
          LOG(FATAL) << "Unexpected data type for auto parallelization: "
                     << event->buffer->GetTypeId().name();
        }
      } else if (event->message_strategy ==
                     MessageStrategy::kAlongEdgeToOuterVertex ||
                 event->message_strategy ==
                     MessageStrategy::kAlongIncomingEdgeToOuterVertex ||
                 event->message_strategy ==
                     MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
        if (event->buffer->GetTypeId() == typeid(double)) {
          syncOnInnerVertexSend<double>(i_ec_frag, event->buffer,
                                        event->event_id,
                                        event->message_strategy);
        } else if (event->buffer->GetTypeId() == typeid(uint32_t)) {
          syncOnInnerVertexSend<uint32_t>(i_ec_frag, event->buffer,
                                          event->event_id,
                                          event->message_strategy);
        } else if (event->buffer->GetTypeId() == typeid(int32_t)) {
          syncOnInnerVertexSend<int32_t>(i_ec_frag, event->buffer,
                                         event->event_id,
                                         event->message_strategy);
        } else if (event->buffer->GetTypeId() == typeid(int64_t)) {
          syncOnInnerVertexSend<int64_t>(i_ec_frag, event->buffer,
                                         event->event_id,
                                         event->message_strategy);
        } else if (event->buffer->GetTypeId() == typeid(uint64_t)) {
          syncOnInnerVertexSend<uint64_t>(i_ec_frag, event->buffer,
                                          event->event_id,
                                          event->message_strategy);
        } else if (event->buffer->GetTypeId() ==
                   typeid(std::vector<uint32_t>)) {
          syncOnInnerVertexSend<std::vector<uint32_t>>(i_ec_frag, event->buffer,
                                                       event->event_id,
                                                       event->message_strategy);
        } else if (event->buffer->GetTypeId() ==
                   typeid(std::vector<uint64_t>)) {
          syncOnInnerVertexSend<std::vector<uint64_t>>(i_ec_frag, event->buffer,
                                                       event->event_id,
                                                       event->message_strategy);
        } else {
          LOG(FATAL) << "Unexpected data type for auto parallelization: "
                     << event->buffer->GetTypeId().name();
        }
      } else {
        LOG(FATAL) << "Unexpected message stratety "
                   << underlying_value(event->message_strategy);
      }
    }
  }

  template <typename T>
  inline void syncOnInnerVertexSend(const FRAG_T& frag, ISyncBuffer* buffer,
                                    int event_id,
                                    MessageStrategy message_strategy) {
    auto* bptr =
        dynamic_cast<SyncBuffer<typename FRAG_T::vertices_t, T>*>(buffer);
    auto inner_vertices = frag.InnerVertices();
    std::vector<size_t> message_num(Base::fnum(), 0);

    if (message_strategy == MessageStrategy::kAlongEdgeToOuterVertex) {
      for (auto v : inner_vertices) {
        if (bptr->IsUpdated(v)) {
          auto dsts = frag.IOEDests(v);
          const fid_t* ptr = dsts.begin;
          while (ptr != dsts.end) {
            ++message_num[*(ptr++)];
          }
        }
      }
    } else if (message_strategy ==
               MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      for (auto v : inner_vertices) {
        if (bptr->IsUpdated(v)) {
          auto dsts = frag.IEDests(v);
          const fid_t* ptr = dsts.begin;
          while (ptr != dsts.end) {
            ++message_num[*(ptr++)];
          }
        }
      }
    } else if (message_strategy ==
               MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      for (auto v : inner_vertices) {
        if (bptr->IsUpdated(v)) {
          auto dsts = frag.OEDests(v);
          const fid_t* ptr = dsts.begin;
          while (ptr != dsts.end) {
            ++message_num[*(ptr++)];
          }
        }
      }
    }

    for (fid_t i = 0; i < Base::fnum(); i++) {
      if (message_num[i] > 0) {
        Base::SendToFragment<int>(i, event_id);
        Base::SendToFragment<size_t>(i, message_num[i]);
      }
    }

    if (message_strategy == MessageStrategy::kAlongEdgeToOuterVertex) {
      for (auto v : inner_vertices) {
        if (bptr->IsUpdated(v)) {
          Base::SendMsgThroughEdges(frag, v, bptr->GetValue(v));
          bptr->Reset(v);
        }
      }
    } else if (message_strategy ==
               MessageStrategy::kAlongIncomingEdgeToOuterVertex) {
      for (auto v : inner_vertices) {
        if (bptr->IsUpdated(v)) {
          Base::SendMsgThroughIEdges(frag, v, bptr->GetValue(v));
          bptr->Reset(v);
        }
      }
    } else if (message_strategy ==
               MessageStrategy::kAlongOutgoingEdgeToOuterVertex) {
      for (auto v : inner_vertices) {
        if (bptr->IsUpdated(v)) {
          Base::SendMsgThroughOEdges(frag, v, bptr->GetValue(v));
          bptr->Reset(v);
        }
      }
    }
  }

  template <typename T>
  inline void syncOnOuterVertexSend(const FRAG_T& frag, ISyncBuffer* buffer,
                                    int event_id) {
    auto* bptr =
        dynamic_cast<SyncBuffer<typename FRAG_T::vertices_t, T>*>(buffer);
    auto inner_vertices = frag.InnerVertices();
    auto outer_vertices = frag.OuterVertices();
    std::vector<size_t> message_num(Base::fnum(), 0);

    for (auto v : inner_vertices) {
      bptr->Reset(v);
    }

    for (auto v : outer_vertices) {
      if (bptr->IsUpdated(v)) {
        fid_t fid = frag.GetFragId(v);
        ++message_num[fid];
      }
    }

    for (fid_t i = 0; i < Base::fnum(); i++) {
      if (message_num[i] > 0) {
        Base::SendToFragment<int>(i, event_id);
        Base::SendToFragment<size_t>(i, message_num[i]);
      }
    }

    for (auto v : outer_vertices) {
      if (bptr->IsUpdated(v)) {
        Base::SyncStateOnOuterVertex(frag, v, bptr->GetValue(v));
        bptr->Reset(v);
      }
    }
  }

  template <typename T>
  inline void syncOnVertexRecv(const FRAG_T& frag, ISyncBuffer* buffer) {
    auto* bptr =
        dynamic_cast<SyncBuffer<typename FRAG_T::vertices_t, T>*>(buffer);

    size_t message_num = 0;
    T rhs;
    Vertex<vid_t> v(0);
    Base::GetMessage<size_t>(message_num);
    while (message_num--) {
      GetMessage(frag, v, rhs);
      bptr->Aggregate(v, std::move(rhs));
    }
  }

  std::vector<ap_event> auto_parallel_events_;
};  // namespace grape

}  // namespace grape

#endif  // GRAPE_PARALLEL_AUTO_PARALLEL_MESSAGE_MANAGER_H_
