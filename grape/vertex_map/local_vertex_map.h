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

#ifndef GRAPE_VERTEX_MAP_LOCAL_VERTEX_MAP_H_
#define GRAPE_VERTEX_MAP_LOCAL_VERTEX_MAP_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grape/config.h"
#include "grape/fragment/partitioner.h"
#include "grape/graph/id_indexer.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/vertex_map/vertex_map_base.h"
#include "grape/worker/comm_spec.h"

namespace grape {

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class LocalVertexMap;

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class LocalVertexMapBuilder {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 private:
  LocalVertexMapBuilder(
      fid_t fid, std::vector<IdIndexer<internal_oid_t, VID_T>>& oid_to_index,
      std::vector<IdIndexer<VID_T, VID_T>>& gid_to_index,
      const PARTITIONER_T& partitioner, const IdParser<VID_T>& id_parser)
      : fid_(fid),
        oid_to_index_(oid_to_index),
        gid_to_index_(gid_to_index),
        partitioner_(partitioner),
        id_parser_(id_parser) {}

 public:
  ~LocalVertexMapBuilder() {}

  void add_local_vertex(const internal_oid_t& id, VID_T& gid) {
    assert(partitioner_.GetPartitionId(id) == fid_);
    oid_to_index_[fid_].add(id, gid);
    gid = id_parser_.generate_global_id(fid_, gid);
  }

  void add_vertex(const internal_oid_t& id) {
    fid_t fid = partitioner_.GetPartitionId(id);
    oid_to_index_[fid]._add(id);
  }

  void finish(LocalVertexMap<OID_T, VID_T, PARTITIONER_T>& vertex_map) {
    const CommSpec& comm_spec = vertex_map.GetCommSpec();
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    std::thread request_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker_id = (worker_id + i) % worker_num;
        auto& indexer = oid_to_index_[comm_spec.WorkerToFrag(dst_worker_id)];
        sync_comm::Send(indexer.keys(), dst_worker_id, 0, comm_spec.comm());
        std::vector<VID_T> gid_list(indexer.size());
        sync_comm::Recv(gid_list, dst_worker_id, 1, comm_spec.comm());
        auto& gid_indexer =
            gid_to_index_[comm_spec.WorkerToFrag(dst_worker_id)];
        for (auto gid : gid_list) {
          gid_indexer._add(gid);
        }
      }
    });
    std::thread response_thread([&]() {
      for (int i = 1; i < worker_num; ++i) {
        int src_worker_id = (worker_id + worker_num - i) % worker_num;
        typename IdIndexer<internal_oid_t, VID_T>::key_buffer_t keys;
        sync_comm::Recv(keys, src_worker_id, 0, comm_spec.comm());
        std::vector<VID_T> gid_list(keys.size());
        VID_T gid;
        auto& native_indexer = oid_to_index_[fid_];
        for (size_t k = 0; k < keys.size(); ++k) {
          CHECK(native_indexer.get_index(keys[k], gid));
          gid = id_parser_.generate_global_id(fid_, gid);
          gid_list[k] = gid;
        }
        sync_comm::Send(gid_list, src_worker_id, 1, comm_spec.comm());
      }
    });

    request_thread.join();
    response_thread.join();
    MPI_Barrier(comm_spec.comm());

    vertex_map.vertices_num_.resize(comm_spec.fnum());
    vertex_map.vertices_num_[fid_] = oid_to_index_[fid_].size();
    sync_comm::AllGather(vertex_map.vertices_num_, comm_spec.comm());
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class LocalVertexMap;

  fid_t fid_;
  std::vector<IdIndexer<internal_oid_t, VID_T>>& oid_to_index_;
  std::vector<IdIndexer<VID_T, VID_T>>& gid_to_index_;
  const PARTITIONER_T& partitioner_;
  const IdParser<VID_T> id_parser_;
};

template <typename OID_T, typename VID_T,
          typename PARTITIONER_T = HashPartitioner<OID_T>>
class LocalVertexMap : public VertexMapBase<OID_T, VID_T, PARTITIONER_T> {
  using base_t = VertexMapBase<OID_T, VID_T, PARTITIONER_T>;
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  explicit LocalVertexMap(const CommSpec& comm_spec) : base_t(comm_spec) {}
  ~LocalVertexMap() = default;
  void Init() {
    oid_to_index_.resize(comm_spec_.fnum());
    gid_to_index_.resize(comm_spec_.fnum());
  }

  size_t GetTotalVertexSize() const {
    size_t size = 0;
    for (auto v : vertices_num_) {
      size += v;
    }
    return size;
  }

  size_t GetInnerVertexSize(fid_t fid) const { return vertices_num_[fid]; }
  void AddVertex(const OID_T& oid) { LOG(FATAL) << "not implemented"; }

  using base_t::Lid2Gid;
  bool AddVertex(const OID_T& oid, VID_T& gid) {
    LOG(FATAL) << "not implemented";
    return false;
  }

  using base_t::GetFidFromGid;
  using base_t::GetLidFromGid;
  bool GetOid(const VID_T& gid, OID_T& oid) const {
    fid_t fid = GetFidFromGid(gid);
    return GetOid(fid, id_parser_.get_local_id(gid), oid);
  }

  bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) const {
    internal_oid_t internal_oid;
    if (fid == comm_spec_.fid()) {
      if (oid_to_index_[fid].get_key(lid, internal_oid)) {
        oid = InternalOID<OID_T>::FromInternal(internal_oid);
        return true;
      }
    } else {
      VID_T index;
      if (gid_to_index_[fid].get_index(id_parser_.generate_global_id(fid, lid),
                                       index)) {
        if (oid_to_index_[fid].get_key(index, internal_oid)) {
          oid = InternalOID<OID_T>::FromInternal(internal_oid);
          return true;
        }
      }
    }
    return false;
  }

  bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) const {
    internal_oid_t internal_oid(oid);
    return _GetGid(fid, internal_oid, gid);
  }

  bool _GetGid(fid_t fid, const internal_oid_t& oid, VID_T& gid) const {
    VID_T index;
    if (fid == comm_spec_.fid()) {
      if (oid_to_index_[fid].get_index(oid, index)) {
        gid = id_parser_.generate_global_id(fid, index);
        return true;
      }
    } else {
      if (oid_to_index_[fid].get_index(oid, index)) {
        return gid_to_index_[fid].get_key(index, gid);
      }
    }
    return false;
  }

  bool GetGid(const OID_T& oid, VID_T& gid) const {
    fid_t fid = partitioner_.GetPartitionId(oid);
    return GetGid(fid, oid, gid);
  }

  bool _GetGid(const internal_oid_t& oid, VID_T& gid) const {
    fid_t fid = partitioner_.GetPartitionId(oid);
    return _GetGid(fid, oid, gid);
  }

  LocalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T> GetLocalBuilder() {
    fid_t fid = comm_spec_.fid();
    return LocalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T>(
        fid, oid_to_index_, gid_to_index_, partitioner_, id_parser_);
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s_%d", prefix.c_str(),
             kSerializationVertexMapFilename, comm_spec_.fid());

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    base_t::serialize(io_adaptor);
    for (auto& indexer : oid_to_index_) {
      indexer.Serialize(io_adaptor);
    }
    for (auto& indexer : gid_to_index_) {
      indexer.Serialize(io_adaptor);
    }
    io_adaptor->Close();
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& prefix, fid_t fid) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s_%d", prefix.c_str(),
             kSerializationVertexMapFilename, fid);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    base_t::deserialize(io_adaptor);
    oid_to_index_.resize(comm_spec_.fnum());
    for (auto& indexer : oid_to_index_) {
      indexer.Deserialize(io_adaptor);
    }
    gid_to_index_.resize(comm_spec_.fnum());
    for (auto& indexer : gid_to_index_) {
      indexer.Deserialize(io_adaptor);
    }
    io_adaptor->Close();
  }

  void UpdateToBalance(std::vector<VID_T>& vnum_list,
                       std::vector<std::vector<VID_T>>& gid_maps) {
    LOG(FATAL) << "not implemented";
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class LocalVertexMapBuilder;

  std::vector<IdIndexer<internal_oid_t, VID_T>> oid_to_index_;
  std::vector<IdIndexer<VID_T, VID_T>> gid_to_index_;
  using base_t::comm_spec_;
  using base_t::id_parser_;
  using base_t::partitioner_;

  std::vector<VID_T> vertices_num_;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_LOCAL_VERTEX_MAP_H_
