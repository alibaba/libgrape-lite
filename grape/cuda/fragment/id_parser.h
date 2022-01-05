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

#ifndef GRAPE_CUDA_FRAGMENT_ID_PARSER_H_
#define GRAPE_CUDA_FRAGMENT_ID_PARSER_H_

#include "grape/config.h"
#include "grape/cuda/utils/cuda_utils.h"

namespace grape {
template <typename VID_T>
class IdParser {
 public:
  DEV_HOST_INLINE void Init(fid_t fnum) {
    fnum_ = fnum;
    fid_t maxfid = fnum_ - 1;
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

  DEV_HOST_INLINE fid_t GetFid(VID_T gid) const { return (gid >> fid_offset_); }

  DEV_HOST_INLINE VID_T GetLid(VID_T gid) const { return (gid & id_mask_); }

  DEV_HOST_INLINE VID_T Lid2Gid(fid_t fid, VID_T lid) const {
    VID_T gid = fid;
    gid = (gid << fid_offset_) | lid;
    return gid;
  }

 private:
  fid_t fnum_;
  VID_T fid_offset_;
  VID_T id_mask_;
};
}  // namespace grape

#endif  // GRAPE_CUDA_FRAGMENT_ID_PARSER_H_
