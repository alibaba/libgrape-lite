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
#include "grape/fragment/id_parser.h"
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
template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class VertexMapBase {
 public:
  using partitioner_t = PARTITIONER_T;
  using oid_t = OID_T;
  using vid_t = VID_T;
  explicit VertexMapBase(const CommSpec& comm_spec)
      : comm_spec_(comm_spec), partitioner_() {
    comm_spec_.Dup();
    id_parser_.init(comm_spec_.fnum());
  }
  virtual ~VertexMapBase() = default;

  void SetPartitioner(const PARTITIONER_T& partitioner) {
    partitioner_ = partitioner;
  }

  void SetPartitioner(PARTITIONER_T&& partitioner) {
    partitioner_ = std::move(partitioner);
  }

  fid_t GetFragmentNum() const { return comm_spec_.fnum(); }

  VID_T Lid2Gid(fid_t fid, const VID_T& lid) const {
    return id_parser_.generate_global_id(fid, lid);
  }

  fid_t GetFidFromGid(const VID_T& gid) const {
    return id_parser_.get_fragment_id(gid);
  }

  VID_T GetLidFromGid(const VID_T& gid) const {
    return id_parser_.get_local_id(gid);
  }

  VID_T MaxVertexNum() const { return id_parser_.max_local_id(); }

  const CommSpec& GetCommSpec() const { return comm_spec_; }

  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    partitioner_.template serialize<IOADAPTOR_T>(writer);
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    id_parser_.init(comm_spec_.fnum());
    partitioner_.template deserialize<IOADAPTOR_T>(reader);
  }

  fid_t GetFragmentId(const OID_T& oid) const {
    return partitioner_.GetPartitionId(oid);
  }

  const PARTITIONER_T& GetPartitioner() const { return partitioner_; }

  PARTITIONER_T& GetPartitioner() { return partitioner_; }

 protected:
  CommSpec comm_spec_;
  PARTITIONER_T partitioner_;
  IdParser<VID_T> id_parser_;

 public:
  // get metadata of the graph.
  virtual size_t GetTotalVertexSize() const = 0;
  virtual size_t GetInnerVertexSize(fid_t fid) const = 0;

  // for constructing the vertexmap.
  virtual void AddVertex(const OID_T& oid) = 0;
  virtual bool AddVertex(const OID_T& oid, VID_T& gid) = 0;

  virtual void UpdateToBalance(std::vector<VID_T>& vnum_list,
                               std::vector<std::vector<VID_T>>& gid_maps) = 0;

  // convert the vertex ids with the help of mappings.
  virtual bool GetOid(const VID_T& gid, OID_T& oid) const = 0;

  virtual bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) const = 0;

  virtual bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) const = 0;

  virtual bool GetGid(const OID_T& oid, VID_T& gid) const = 0;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_VERTEX_MAP_BASE_H_
