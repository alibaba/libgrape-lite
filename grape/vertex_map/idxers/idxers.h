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

#ifndef GRAPE_VERTEX_MAP_IDXERS_IDXERS_H_
#define GRAPE_VERTEX_MAP_IDXERS_IDXERS_H_

#include "grape/vertex_map/idxers/hashmap_idxer.h"
#include "grape/vertex_map/idxers/hashmap_idxer_view.h"
#include "grape/vertex_map/idxers/local_idxer.h"
#include "grape/vertex_map/idxers/pthash_idxer.h"

namespace grape {

template <typename OID_T, typename VID_T>
IdxerBase<OID_T, VID_T>* deserialize_idxer(
    std::unique_ptr<IOAdaptorBase>& reader) {
  int type;
  reader->Read(&type, sizeof(type));
  IdxerType idxer_type = static_cast<IdxerType>(type);
  switch (idxer_type) {
  case IdxerType::kHashMapIdxer: {
    auto idxer = new HashMapIdxer<OID_T, VID_T>();
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kLocalIdxer: {
    auto idxer = new LocalIdxer<OID_T, VID_T>();
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kHashMapIdxerView: {
    auto idxer = new HashMapIdxerView<OID_T, VID_T>();
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kPTHashIdxer: {
    auto idxer = new PTHashIdxer<OID_T, VID_T>();
    idxer->deserialize(reader);
    return idxer;
  }
  default:
    return nullptr;
  }
}

template <typename OID_T, typename VID_T>
IdxerBase<OID_T, VID_T>* extend_indexer(IdxerBase<OID_T, VID_T>* input,
                                        const std::vector<OID_T>& id_list) {
  if (input->type() == IdxerType::kHashMapIdxer) {
    auto casted = dynamic_cast<HashMapIdxer<OID_T, VID_T>*>(input);
    for (auto& id : id_list) {
      casted->add(id);
    }
    return input;
  } else {
    LOG(FATAL) << "Only HashMapIdxer can be extended";
  }
  return nullptr;
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_IDXERS_H_
