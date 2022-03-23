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

#ifndef GRAPE_VERTEX_MAP_GLOBAL_VERTEX_MAP_H_
#define GRAPE_VERTEX_MAP_GLOBAL_VERTEX_MAP_H_

#include <algorithm>
#include <atomic>
#include <fstream>
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
class GlobalVertexMap;

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class GlobalVertexMapBuilder {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 private:
  GlobalVertexMapBuilder(fid_t fid, IdIndexer<internal_oid_t, VID_T>& indexer,
                         const PARTITIONER_T& partitioner,
                         const IdParser<VID_T>& id_parser)
      : fid_(fid),
        indexer_(indexer),
        partitioner_(partitioner),
        id_parser_(id_parser) {}

 public:
  ~GlobalVertexMapBuilder() {}

  void add_local_vertex(const internal_oid_t& id, VID_T& gid) {
    assert(partitioner_.GetPartitionId(id) == fid_);
    indexer_.add(id, gid);
    id_parser_.generate_global_id(fid_, gid);
  }

  void add_vertex(const internal_oid_t& id) {
    if (partitioner_.GetPartitionId(id) == fid_) {
      indexer_._add(id);
    }
  }

  void finish(GlobalVertexMap<OID_T, VID_T, PARTITIONER_T>& vertex_map) {
    const CommSpec& comm_spec = vertex_map.GetCommSpec();
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    fid_t fnum = comm_spec.fnum();
    {
      std::thread recv_thread([&]() {
        int src_worker_id = (worker_id + 1) % worker_num;
        while (src_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum; ++fid) {
            if (comm_spec.FragToWorker(fid) != src_worker_id) {
              continue;
            }
            sync_comm::Recv(vertex_map.indexers_[fid], src_worker_id, 0,
                            comm_spec.comm());
          }
          src_worker_id = (src_worker_id + 1) % worker_num;
        }
      });
      std::thread send_thread([&]() {
        int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
        while (dst_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum; ++fid) {
            if (comm_spec.FragToWorker(fid) != worker_id) {
              continue;
            }
            sync_comm::Send(indexer_, dst_worker_id, 0, comm_spec.comm());
          }
          dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
        }
      });
      send_thread.join();
      recv_thread.join();
    }
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class GlobalVertexMap;

  fid_t fid_;
  IdIndexer<internal_oid_t, VID_T>& indexer_;
  const PARTITIONER_T& partitioner_;
  const IdParser<VID_T>& id_parser_;
};

/**
 * @brief a kind of VertexMapBase which holds global mapping information in
 * each worker.
 *
 * @tparam OID_T
 * @tparam VID_T
 */
template <typename OID_T, typename VID_T,
          typename PARTITIONER_T = HashPartitioner<OID_T>>
class GlobalVertexMap : public VertexMapBase<OID_T, VID_T, PARTITIONER_T> {
  // TODO(lxj): to support shared-memory for workers on same host (auto apps)

  using base_t = VertexMapBase<OID_T, VID_T, PARTITIONER_T>;
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  explicit GlobalVertexMap(const CommSpec& comm_spec) : base_t(comm_spec) {}
  ~GlobalVertexMap() = default;
  void Init() { indexers_.resize(comm_spec_.fnum()); }

  size_t GetTotalVertexSize() const {
    size_t size = 0;
    for (const auto& v : indexers_) {
      size += v.size();
    }
    return size;
  }

  size_t GetInnerVertexSize(fid_t fid) const { return indexers_[fid].size(); }
  void AddVertex(const OID_T& oid) {
    fid_t fid = partitioner_.GetPartitionId(oid);
    indexers_[fid]._add(oid);
  }

  using base_t::Lid2Gid;
  bool AddVertex(const OID_T& oid, VID_T& gid) {
    fid_t fid = partitioner_.GetPartitionId(oid);
    internal_oid_t internal_oid(oid);
    if (indexers_[fid].add(std::move(internal_oid), gid)) {
      gid = Lid2Gid(fid, gid);
      return true;
    }
    gid = Lid2Gid(fid, gid);
    return false;
  }

  bool AddVertex(OID_T&& oid, VID_T& gid) {
    fid_t fid = partitioner_.GetPartitionId(oid);
    internal_oid_t internal_oid(std::move(oid));
    if (indexers_[fid].add(std::move(internal_oid), gid)) {
      gid = Lid2Gid(fid, gid);
      return true;
    }
    gid = Lid2Gid(fid, gid);
    return false;
  }

  using base_t::GetFidFromGid;
  using base_t::GetLidFromGid;
  bool GetOid(const VID_T& gid, OID_T& oid) const {
    fid_t fid = GetFidFromGid(gid);
    VID_T lid = GetLidFromGid(gid);
    return GetOid(fid, lid, oid);
  }

  bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) const {
    internal_oid_t internal_oid;
    if (indexers_[fid].get_key(lid, internal_oid)) {
      oid = InternalOID<OID_T>::FromInternal(internal_oid);
      return true;
    }
    return false;
  }

  bool _GetGid(fid_t fid, const internal_oid_t& oid, VID_T& gid) const {
    if (indexers_[fid].get_index(oid, gid)) {
      gid = Lid2Gid(fid, gid);
      return true;
    }
    return false;
  }

  bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) const {
    internal_oid_t internal_oid(oid);
    return _GetGid(fid, internal_oid, gid);
  }

  bool _GetGid(const internal_oid_t& oid, VID_T& gid) const {
    fid_t fid = partitioner_.GetPartitionId(oid);
    return _GetGid(fid, oid, gid);
  }

  bool GetGid(const OID_T& oid, VID_T& gid) const {
    fid_t fid = partitioner_.GetPartitionId(oid);
    return GetGid(fid, oid, gid);
  }

  GlobalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T> GetLocalBuilder() {
    fid_t fid = comm_spec_.fid();
    return GlobalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T>(
        fid, indexers_[fid], partitioner_, id_parser_);
  }

 private:
  template <typename IOADAPTOR_T>
  void serialize(const std::string& path) {
    auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(path));
    io_adaptor->Open("wb");
    base_t::serialize(io_adaptor);
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      indexers_[i].Serialize(io_adaptor);
    }
    io_adaptor->Close();
  }

 public:
  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);
    std::string path = std::string(fbuf);
    if (comm_spec_.worker_id() == 0) {
      serialize<IOADAPTOR_T>(path);
    }
    MPI_Barrier(comm_spec_.comm());
    auto exists_file = [](const std::string& name) {
      std::ifstream f(name.c_str());
      return f.good();
    };
    if (!exists_file(path) && comm_spec_.local_id() == 0) {
      serialize<IOADAPTOR_T>(path);
    }
    MPI_Barrier(comm_spec_.comm());
    if (!exists_file(path)) {
      serialize<IOADAPTOR_T>(path);
    }
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& prefix, fid_t fid) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    base_t::deserialize(io_adaptor);

    indexers_.resize(comm_spec_.fnum());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      indexers_[i].Deserialize(io_adaptor);
    }
    io_adaptor->Close();
  }

  void UpdateToBalance(std::vector<VID_T>& vnum_list,
                       std::vector<std::vector<VID_T>>& gid_maps) {
    fid_t fnum = comm_spec_.fnum();
    std::vector<std::vector<internal_oid_t>> oid_lists(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      oid_lists[i].resize(vnum_list[i]);
    }
    for (fid_t fid = 0; fid < fnum; ++fid) {
      auto& old_indexer = indexers_[fid];
      VID_T vnum = old_indexer.size();
      for (VID_T i = 0; i < vnum; ++i) {
        VID_T new_gid = gid_maps[fid][i];
        internal_oid_t oid;
        fid_t new_fid = GetFidFromGid(new_gid);
        CHECK(old_indexer.get_key(i, oid));
        if (new_fid != fid) {
          OID_T id = InternalOID<OID_T>::FromInternal(oid);
          partitioner_.SetPartitionId(id, new_fid);
        }
        VID_T new_lid = GetLidFromGid(new_gid);
        oid_lists[new_fid][new_lid] = oid;
      }
    }
    std::vector<IdIndexer<internal_oid_t, VID_T>> new_indexers(fnum);
    for (fid_t i = 0; i < fnum; ++i) {
      auto& indexer = new_indexers[i];
      for (auto& oid : oid_lists[i]) {
        indexer._add(oid);
      }
    }
    std::swap(indexers_, new_indexers);
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class GlobalVertexMapBuilder;

  std::vector<IdIndexer<internal_oid_t, VID_T>> indexers_;
  using base_t::comm_spec_;
  using base_t::id_parser_;
  using base_t::partitioner_;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_GLOBAL_VERTEX_MAP_H_
