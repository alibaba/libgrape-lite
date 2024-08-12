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
#include "grape/vertex_map/idxers/sorted_array_idxer.h"

namespace grape {

template <typename OID_T, typename VID_T>
std::unique_ptr<IdxerBase<OID_T, VID_T>> deserialize_idxer(
    std::unique_ptr<IOAdaptorBase>& reader) {
  int type;
  reader->Read(&type, sizeof(type));
  IdxerType idxer_type = static_cast<IdxerType>(type);
  switch (idxer_type) {
  case IdxerType::kHashMapIdxer: {
    auto idxer = std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new HashMapIdxer<OID_T, VID_T>());
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kLocalIdxer: {
    auto idxer = std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new LocalIdxer<OID_T, VID_T>());
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kHashMapIdxerView: {
    auto idxer = std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new HashMapIdxerView<OID_T, VID_T>());
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kPTHashIdxer: {
    auto idxer = std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new PTHashIdxer<OID_T, VID_T>());
    idxer->deserialize(reader);
    return idxer;
  }
  case IdxerType::kSortedArrayIdxer: {
    auto idxer = std::unique_ptr<IdxerBase<OID_T, VID_T>>(
        new SortedArrayIdxer<OID_T, VID_T>());
    idxer->deserialize(reader);
    return idxer;
  }
  default:
    return nullptr;
  }
}

template <typename OID_T, typename VID_T>
std::unique_ptr<IdxerBase<OID_T, VID_T>> extend_indexer(
    std::unique_ptr<IdxerBase<OID_T, VID_T>>&& input,
    const std::vector<OID_T>& id_list, VID_T base) {
  if (input->type() == IdxerType::kHashMapIdxer) {
    auto casted = std::unique_ptr<HashMapIdxer<OID_T, VID_T>>(
        dynamic_cast<HashMapIdxer<OID_T, VID_T>*>(input.release()));
    for (auto& id : id_list) {
      casted->add(id);
    }
    return casted;
  } else if (input->type() == IdxerType::kLocalIdxer) {
    auto casted = std::unique_ptr<LocalIdxer<OID_T, VID_T>>(
        dynamic_cast<LocalIdxer<OID_T, VID_T>*>(input.release()));
    for (auto& id : id_list) {
      casted->add(id, base++);
    }
    return casted;
  } else {
    LOG(ERROR) << "Only HashMapIdxer or LocalIdxer can be extended";
    return std::move(input);
  }
}

inline IdxerType parse_idxer_type_name(const std::string& name) {
  if (name == "hashmap") {
    return IdxerType::kHashMapIdxer;
  } else if (name == "local") {
    return IdxerType::kLocalIdxer;
  } else if (name == "pthash") {
    return IdxerType::kPTHashIdxer;
  } else if (name == "sorted_array") {
    return IdxerType::kSortedArrayIdxer;
  } else if (name == "hashmap_view") {
    return IdxerType::kHashMapIdxerView;
  } else {
    LOG(INFO) << "unrecognized idxer type: " << name
              << ", use hashmap idxer "
                 "as default";
    return IdxerType::kHashMapIdxer;
  }
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IDXERS_IDXERS_H_
