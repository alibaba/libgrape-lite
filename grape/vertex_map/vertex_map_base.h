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

#ifndef GRAPE_VERTEX_MAP_VERTEX_MAP_BASE_H_
#define GRAPE_VERTEX_MAP_VERTEX_MAP_BASE_H_

#include <string>
#include <vector>

#include "grape/config.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/worker/comm_spec.h"

namespace grape {

/**
 * @brief VertexMapBase manages some mapping about vertices.
 *
 * a <VertexMapBase> manages:
 *
 * 1) which fragment a vertex resides in as a inner_vertex, for edge-cut
 * distributed graphs;
 *
 * 2) which fragment a vertex resides in as a master_vertex,
 * for vertex-cut distributed graphs;
 *
 * 3) the mapping from ids. There are 3 kinds of vertex ids in grape.
 *
 *  - original_id (a.k.a., OID), is provided by the origin dataset, it may be
 * not continoues, or even strings.
 *
 *  - local_id (a.k.a., LID), is allocated WITHIN a fragment, it is continoues
 * and increased from 1.
 *
 *  - global_id (a.k.a., GID), is unique in the distributed graph and works as
 * the identifier of a vertex in libgrape-lite. It consists of two parts and
 * formatted as fid|local_id.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the derived
 * classes would be invoked directly, not via virtual functions.
 *
 * @tparam OID_T
 * @tparam VID_T
 */
template <typename OID_T, typename VID_T>
class VertexMapBase {
 public:
  explicit VertexMapBase(const CommSpec& comm_spec) : comm_spec_(comm_spec) {}
  virtual ~VertexMapBase() = default;

  virtual void Init() {
    fnum_ = comm_spec_.fnum();
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

  fid_t GetFragmentNum() const { return fnum_; }

  VID_T Lid2Gid(fid_t fid, const VID_T& lid) const {
    VID_T gid = fid;
    gid = (gid << fid_offset_) | lid;
    return gid;
  }

  fid_t GetFidFromGid(const VID_T& gid) const {
    return (fid_t)(gid >> fid_offset_);
  }

  VID_T GetLidFromGid(const VID_T& gid) const { return (gid & id_mask_); }

  VID_T MaxVertexNum() const { return (static_cast<VID_T>(1) << fid_offset_); }

  const CommSpec& GetCommSpec() const { return comm_spec_; }

  void BaseSerialize(InArchive& in_archive) const {
    in_archive << fnum_ << fid_offset_ << id_mask_;
  }

  void BaseDeserialize(OutArchive& out_archive) {
    out_archive >> fnum_ >> fid_offset_ >> id_mask_;
    CHECK_EQ(comm_spec_.fnum(), fnum_);
  }

  int GetFidOffset() const { return fid_offset_; }

 protected:
  fid_t fnum_;
  int fid_offset_;
  VID_T id_mask_;
  CommSpec comm_spec_;

 public:
  // get metadata of the graph.
  virtual size_t GetTotalVertexSize() = 0;
  virtual size_t GetInnerVertexSize(fid_t fid) = 0;

  // for constructing the vertexmap.
  virtual void Clear() = 0;
  virtual void AddVertex(fid_t fid, const OID_T& oid) = 0;
  virtual bool AddVertex(fid_t fid, const OID_T& oid, VID_T& gid) = 0;
  virtual void Construct() = 0;

  virtual void UpdateToBalance(std::vector<VID_T>& vnum_list,
                               std::vector<std::vector<VID_T>>& gid_maps) = 0;

  // convert the vertex ids with the help of mappings.
  virtual bool GetOid(const VID_T& vid, OID_T& oid) = 0;

  virtual bool GetOid(fid_t fid, const VID_T& vid, OID_T& oid) = 0;

  virtual bool GetGid(const OID_T& oid, VID_T& gid) = 0;

  virtual bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) = 0;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_VERTEX_MAP_BASE_H_
