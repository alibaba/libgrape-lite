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

#ifndef GRAPE_VERTEX_MAP_IDXERS_IDXER_BASE_H_
#define GRAPE_VERTEX_MAP_IDXERS_IDXER_BASE_H_

#include "grape/worker/comm_spec.h"

namespace grape {

enum class IdxerType {
  kHashMapIdxer,
  kLocalIdxer,
  kPTHashIdxer,
  kHashMapIdxerView,
  kSortedArrayIdxer,
};

template <typename OID_T, typename VID_T>
class IdxerBase {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  virtual ~IdxerBase() = default;

  virtual bool get_key(VID_T vid, internal_oid_t& oid) const = 0;

  virtual bool get_index(const internal_oid_t& oid, VID_T& vid) const = 0;

  virtual IdxerType type() const = 0;

  virtual size_t size() const = 0;

  virtual size_t memory_usage() const = 0;

  virtual void serialize(std::unique_ptr<IOAdaptorBase>& writer) = 0;
  virtual void deserialize(std::unique_ptr<IOAdaptorBase>& reader) = 0;
};

template <typename OID_T, typename VID_T>
class IdxerBuilderBase {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  virtual ~IdxerBuilderBase() = default;

  virtual void add(const internal_oid_t& oid) = 0;

  virtual IdxerBase<OID_T, VID_T>* finish() = 0;

  virtual void sync_request(const CommSpec& comm_spec, int target, int tag) = 0;
  virtual void sync_response(const CommSpec& comm_spec, int source,
                             int tag) = 0;
};

template <typename OID_T, typename VID_T>
void serialize_idxer(std::unique_ptr<IOAdaptorBase>& writer,
                     IdxerBase<OID_T, VID_T>* idxer) {
  int type = static_cast<int>(idxer->type());
  writer->Write(&type, sizeof(type));
  idxer->serialize(writer);
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_IDXER_BASE_H_
