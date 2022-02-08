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

#ifndef GRAPE_FRAGMENT_ID_PARSER_H_
#define GRAPE_FRAGMENT_ID_PARSER_H_

#include "grape/config.h"

namespace grape {

template <typename VID_T>
class IdParser {
 public:
  IdParser() = default;

  DEV_HOST_INLINE void init(fid_t fnum) {
    fid_t maxfid = fnum - 1;
    if (maxfid == 0) {
      fid_offset_ = (sizeof(VID_T) * 8) - 1;
    } else {
      int i = 0;
      while (maxfid) {
        maxfid >>= 1;
        ++i;
      }
      fid_offset_ = (sizeof(VID_T) * 8) - i;
    }
    id_mask_ = ((VID_T) 1 << fid_offset_) - (VID_T) 1;
  }

  DEV_HOST_INLINE VID_T max_local_id() const { return id_mask_; }

  DEV_HOST_INLINE VID_T get_local_id(VID_T global_id) const {
    return (global_id & id_mask_);
  }

  DEV_HOST_INLINE fid_t get_fragment_id(VID_T global_id) const {
    return global_id >> fid_offset_;
  }

  DEV_HOST_INLINE VID_T generate_global_id(fid_t fid, VID_T local_id) const {
    return local_id | (static_cast<VID_T>(fid) << fid_offset_);
  }

 private:
  VID_T id_mask_;
  int fid_offset_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_ID_PARSER_H_
