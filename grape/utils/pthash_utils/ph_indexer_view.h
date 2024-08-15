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

#ifndef GRAPE_UTILS_PTHASH_UTILS_PH_INDEXER_VIEW_H_
#define GRAPE_UTILS_PTHASH_UTILS_PH_INDEXER_VIEW_H_

#include "grape/graph/id_indexer.h"
#include "grape/utils/pthash_utils/single_phf_view.h"
#include "grape/utils/ref_vector.h"

namespace grape {

template <typename KEY_T, typename INDEX_T>
class PHIndexerView {
 public:
  PHIndexerView() {}
  ~PHIndexerView() {}

  void init(const void* buffer, size_t size) {
    buffer_ = buffer;
    buffer_size_ = size;

    mem_loader loader(reinterpret_cast<const char*>(buffer), size);
    phf_view_.load(loader);
    keys_view_.load(loader);
  }

  size_t entry_num() const { return keys_view_.size(); }

  bool empty() const { return keys_view_.empty(); }

  bool get_key(INDEX_T lid, KEY_T& oid) const {
    if (lid >= keys_view_.size()) {
      return false;
    }
    oid = keys_view_.get(lid);
    return true;
  }

  bool get_index(const KEY_T& oid, INDEX_T& lid) const {
    auto idx = phf_view_(oid);
    if (idx < keys_view_.size() && keys_view_.get(idx) == oid) {
      lid = idx;
      return true;
    }
    return false;
  }

  size_t size() const { return keys_view_.size(); }

  template <typename IOADAPTOR_T>
  void Serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    writer->Write(&buffer_size_, sizeof(size_t));
    if (buffer_size_ > 0) {
      writer->Write(const_cast<void*>(buffer_), buffer_size_);
    }
  }

 private:
  SinglePHFView<murmurhasher> phf_view_;
  id_indexer_impl::KeyBufferView<KEY_T> keys_view_;

  const void* buffer_;
  size_t buffer_size_;
};

}  // namespace grape

#endif  // GRAPE_UTILS_PTHASH_UTILS_PH_INDEXER_VIEW_H_
