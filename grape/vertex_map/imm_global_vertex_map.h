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

#ifndef GRAPE_VERTEX_MAP_IMM_GLOBAL_VERTEX_MAP_H_
#define GRAPE_VERTEX_MAP_IMM_GLOBAL_VERTEX_MAP_H_

#include "grape/fragment/partitioner.h"
#include "grape/graph/hashmap_indexer.h"
#include "grape/graph/hashmap_indexer_impl.h"
#include "grape/vertex_map/vertex_map_base.h"

namespace grape {

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class ImmGlobalVertexMap;

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class ImmGlobalVertexMapBuilder {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 private:
  ImmGlobalVertexMapBuilder(fid_t fid, const PARTITIONER_T& partitioner,
                            const IdParser<VID_T>& id_parser)
      : fid_(fid), partitioner_(partitioner), id_parser_(id_parser) {}

 public:
  ~ImmGlobalVertexMapBuilder() {}

  void add_vertex(const internal_oid_t& id) {
    if (partitioner_.GetPartitionId(id) == fid_) {
      idxer_._add(id);
    }
  }

  void finish(ImmGlobalVertexMap<OID_T, VID_T, PARTITIONER_T>& vertex_map) {
    std::vector<char> idxer_buf;
    idxer_.serialize_to_mem(idxer_buf);

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
            std::vector<char> buf;
            sync_comm::Recv(buf, src_worker_id, 0, comm_spec.comm());
            vertex_map.indexers_[fid].Init(std::move(buf));
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
            sync_comm::Send(idxer_buf, dst_worker_id, 0, comm_spec.comm());
          }
          dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
        }
      });
      send_thread.join();
      recv_thread.join();

      vertex_map.indexers_[fid_].Init(std::move(idxer_buf));
    }
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class ImmGlobalVertexMap;

  fid_t fid_;
  const PARTITIONER_T& partitioner_;
  const IdParser<VID_T>& id_parser_;

  HMIdxer<internal_oid_t, VID_T> idxer_;
};

template <typename OID_T, typename VID_T,
          typename PARTITIONER_T = HashPartitioner<OID_T>>
class ImmGlobalVertexMap : public VertexMapBase<OID_T, VID_T, PARTITIONER_T> {
  using base_t = VertexMapBase<OID_T, VID_T, PARTITIONER_T>;
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  explicit ImmGlobalVertexMap(const CommSpec& comm_spec) : base_t(comm_spec) {}
  ~ImmGlobalVertexMap() = default;

  void Init() { indexers_.resize(comm_spec_.fnum()); }

  size_t GetTotalVertexSize() const {
    size_t size = 0;
    for (auto& idxer : indexers_) {
      size += idxer.size();
    }
    return size;
  }

  size_t GetInnerVertexSize(fid_t fid) const { return indexers_[fid].size(); }

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

  using base_t::Lid2Gid;
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

  ImmGlobalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T> GetLocalBuilder() {
    fid_t fid = comm_spec_.fid();
    return ImmGlobalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T>(
        fid, partitioner_, id_parser_);
  }

  template <typename IOADAPTOR_T>
  void serialize(const std::string& path) {
    {
      auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(path));
      io_adaptor->Open("wb");
      base_t::serialize(io_adaptor);
      io_adaptor->Close();
    }
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      std::string part_path = path + "_p_" + std::to_string(i);
      indexers_[i].template Serialize<IOADAPTOR_T>(part_path);
    }
  }

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
    {
      auto io_adaptor =
          std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
      io_adaptor->Open();

      base_t::deserialize(io_adaptor);
      io_adaptor->Close();
    }

    indexers_.resize(comm_spec_.fnum());
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      std::string part_path = std::string(fbuf) + "_p_" + std::to_string(i);
      indexers_[i].template Deserialize<IOADAPTOR_T>(part_path);
    }
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class ImmGlobalVertexMapBuilder;

  std::vector<ImmHMIdxer<internal_oid_t, VID_T>> indexers_;
  using base_t::comm_spec_;
  using base_t::id_parser_;
  using base_t::partitioner_;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_IMM_GLOBAL_VERTEX_MAP_H_