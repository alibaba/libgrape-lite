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

#ifndef GRAPE_VERTEX_MAP_VERTEX_MAP_BETA_H_
#define GRAPE_VERTEX_MAP_VERTEX_MAP_BETA_H_

#include <thread>

#include "grape/fragment/id_parser.h"
#include "grape/util.h"
#include "grape/vertex_map/idxer.h"
#include "grape/vertex_map/partitioner.h"

namespace grape {

template <typename OID_T, typename VID_T>
class VertexMapBuilder;

template <typename OID_T, typename VID_T>
class VertexMap {
 public:
  using oid_t = OID_T;
  using vid_t = VID_T;
  using internal_oid_t = typename InternalOID<OID_T>::type;

  VertexMap(const VertexMap&) = delete;
  VertexMap() : partitioner_(NULL) {}
  ~VertexMap() {
    for (auto idxer : idxers_) {
      delete idxer;
    }
    if (partitioner_) {
      delete partitioner_;
    }
  }

  fid_t GetFragmentNum() const { return fnum_; }

  fid_t GetFragmentId(const OID_T& oid) const {
    internal_oid_t internal_oid(oid);
    return partitioner_->GetPartitionId(internal_oid);
  }

  const IdParser<VID_T>& GetIdParser() const { return id_parser_; }

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

  size_t GetTotalVertexSize() const { return total_vertex_size_; }

  size_t GetInnerVertexSize(fid_t fid) const { return inner_vertex_size_[fid]; }

  void UpdateToBalance(const CommSpec& comm_spec,
                       const std::vector<VID_T>& vnum_list,
                       const std::vector<std::vector<VID_T>>& gid_maps);

  bool GetOid(const VID_T& gid, OID_T& oid) const {
    fid_t fid = GetFidFromGid(gid);
    return GetOid(fid, GetLidFromGid(gid), oid);
  }

  bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) const {
    internal_oid_t internal_oid;
    if (idxers_[fid]->get_key(lid, internal_oid)) {
      oid = InternalOID<OID_T>::FromInternal(internal_oid);
      return true;
    }
    return false;
  }

  bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) const {
    internal_oid_t internal_oid(oid);
    if (idxers_[fid]->get_index(internal_oid, gid)) {
      gid = Lid2Gid(fid, gid);
      return true;
    }
    return false;
  }

  bool GetGid(const OID_T& oid, VID_T& gid) const {
    fid_t fid = partitioner_->GetPartitionId(oid);
    return GetGid(fid, oid, gid);
  }

  void reset() {
    for (auto idxer : idxers_) {
      if (idxer) {
        delete idxer;
      }
    }
    idxers_.clear();
  }

  void ExtendVertices(const CommSpec& comm_spec,
                      std::vector<OID_T>&& local_vertices_to_add) {
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    if (is_global_) {
      DistinctSort(local_vertices_to_add);
      std::thread send_thread([&]() {
        int dst_worker_id = (worker_id + 1) % worker_num;
        while (dst_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum_; ++fid) {
            if (comm_spec.FragToWorker(fid) != worker_id) {
              continue;
            }
            sync_comm::Send(local_vertices_to_add, dst_worker_id, 0,
                            comm_spec.comm());
          }
          dst_worker_id = (dst_worker_id + 1) % worker_num;
        }
      });
      std::thread recv_thread([&]() {
        int src_worker_id = (worker_id + worker_num - 1) % worker_num;
        while (src_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum_; ++fid) {
            if (comm_spec.FragToWorker(fid) != src_worker_id) {
              continue;
            }
            std::vector<OID_T> remote_vertices;
            sync_comm::Recv(remote_vertices, src_worker_id, 0,
                            comm_spec.comm());
            idxers_[fid] = extend_indexer(idxers_[fid], remote_vertices);
          }
          src_worker_id = (src_worker_id + worker_num - 1) % worker_num;
        }
      });
      recv_thread.join();
      send_thread.join();
    } else {
      LOG(FATAL) << "Cannot extend vertices in local mode.";
    }
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix, const CommSpec& comm_spec) {
    if (is_global_) {
      char fbuf[1024];
      snprintf(fbuf, sizeof(fbuf), "%s/%s", prefix.c_str(),
               kSerializationVertexMapFilename);
      std::string path = std::string(fbuf);
      if (comm_spec.worker_id() == 0) {
        serialize_impl<IOADAPTOR_T>(path);
      }
      MPI_Barrier(comm_spec.comm());
      if (!exists_file(path) && comm_spec.local_id() == 0) {
        serialize_impl<IOADAPTOR_T>(path);
      }
      MPI_Barrier(comm_spec.comm());
      if (!exists_file(path)) {
        serialize_impl<IOADAPTOR_T>(path);
      }
    } else {
      char fbuf[1024];
      snprintf(fbuf, sizeof(fbuf), "%s/%s_%d", prefix.c_str(),
               kSerializationVertexMapFilename, comm_spec.fid());
      serialize_impl<IOADAPTOR_T>(std::string(fbuf));
    }
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& prefix, const CommSpec& comm_spec) {
    char local_fbuf[1024];
    snprintf(local_fbuf, sizeof(local_fbuf), "%s/%s_%d", prefix.c_str(),
             kSerializationVertexMapFilename, comm_spec.fid());
    if (exists_file(local_fbuf)) {
      deserialize_impl<IOADAPTOR_T>(std::string(local_fbuf));
    } else {
      char global_fbuf[1024];
      snprintf(global_fbuf, sizeof(global_fbuf), "%s/%s", prefix.c_str(),
               kSerializationVertexMapFilename);
      if (exists_file(global_fbuf)) {
        deserialize_impl<IOADAPTOR_T>(std::string(global_fbuf));
      } else {
        LOG(FATAL) << "Cannot find vertex map file.";
      }
    }

    id_parser_.init(fnum_);
  }

  VertexMap& operator=(VertexMap&& other) {
    if (this == &other) {
      return *this;
    }

    this->fnum_ = other.fnum_;
    this->is_global_ = other.is_global_;
    this->total_vertex_size_ = other.total_vertex_size_;
    this->inner_vertex_size_ = std::move(other.inner_vertex_size_);

    this->idxers_ = std::move(other.idxers_);
    this->partitioner_ = other.partitioner_;
    this->id_parser_.init(fnum_);

    other.idxers_.clear();
    other.partitioner_ = nullptr;
    other.total_vertex_size_ = 0;
    other.inner_vertex_size_.clear();

    return *this;
  }

 private:
  template <typename IOADAPTOR>
  void serialize_impl(const std::string& path) {
    auto io_adaptor = new IOADAPTOR(path);
    io_adaptor->Open("wb");
    InArchive arc;
    arc << fnum_ << total_vertex_size_ << inner_vertex_size_;
    io_adaptor->WriteArchive(arc);
    for (fid_t fid = 0; fid < fnum_; ++fid) {
      serialize_idxer<OID_T, VID_T>(io_adaptor, idxers_[fid]);
    }
    serialize_partitioner<OID_T>(io_adaptor, partitioner_);
    delete io_adaptor;
  }

  template <typename IOADAPTOR>
  void deserialize_impl(const std::string& path) {
    auto io_adaptor = new IOADAPTOR(path);
    io_adaptor->Open();
    OutArchive arc;
    io_adaptor->ReadArchive(arc);
    arc >> fnum_ >> total_vertex_size_ >> inner_vertex_size_;
    idxers_.resize(fnum_);
    for (fid_t fid = 0; fid < fnum_; ++fid) {
      idxers_[fid] = deserialize_idxer<OID_T, VID_T>(io_adaptor);
    }
    partitioner_ = deserialize_partitioner<OID_T>(io_adaptor);

    delete io_adaptor;
  }

  template <typename _OID_T, typename _VID_T>
  friend class VertexMapBuilder;

  fid_t fnum_;
  bool is_global_;

  size_t total_vertex_size_;
  std::vector<size_t> inner_vertex_size_;

  std::vector<IdxerBase<OID_T, VID_T>*> idxers_;
  IPartitioner<OID_T>* partitioner_;
  IdParser<VID_T> id_parser_;
};

template <typename OID_T, typename VID_T>
class VertexMapBuilder {
  using internal_oid_t = typename InternalOID<OID_T>::type;

 public:
  VertexMapBuilder(fid_t fid, fid_t fnum, IPartitioner<OID_T>* partitioner,
                   bool is_global = true)
      : fid_(fid), is_global_(is_global), partitioner_(partitioner) {
    idxer_builders_.resize(fnum, nullptr);
    if (!is_global_) {
      for (fid_t i = 1; i < fnum; ++i) {
        idxer_builders_[(i + fid_) % fnum] =
            new LocalIdxerBuilder<OID_T, VID_T>();
      }
    } else {
      for (fid_t i = 1; i < fnum; ++i) {
        idxer_builders_[(i + fid_) % fnum] =
            new DummyIdxerBuilder<OID_T, VID_T>();
      }
    }
    idxer_builders_[fid_] = new HashMapIdxerBuilder<OID_T, VID_T>();
  }

  ~VertexMapBuilder() {
    if (partitioner_) {
      delete partitioner_;
    }
    for (auto idxer_builder : idxer_builders_) {
      if (idxer_builder) {
        delete idxer_builder;
      }
    }
  }

  void add_vertex(const internal_oid_t& id) {
    fid_t fid = partitioner_->GetPartitionId(id);
    idxer_builders_[fid]->add(id);
  }

  void finish(const CommSpec& comm_spec, VertexMap<OID_T, VID_T>& vertex_map) {
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    fid_t fnum = comm_spec.fnum();
    {
      std::thread response_thread = std::thread([&]() {
        int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
        while (dst_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum; ++fid) {
            if (comm_spec.FragToWorker(fid) != worker_id) {
              continue;
            }
            idxer_builders_[fid]->sync_response(comm_spec, dst_worker_id, 0);
          }
          dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
        }
      });
      std::thread request_thread = std::thread([&]() {
        int src_worker_id = (worker_id + 1) % worker_num;
        while (src_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum; ++fid) {
            if (comm_spec.FragToWorker(fid) != src_worker_id) {
              continue;
            }
            idxer_builders_[fid]->sync_request(comm_spec, src_worker_id, 0);
          }
          src_worker_id = (src_worker_id + 1) % worker_num;
        }
      });

      request_thread.join();
      response_thread.join();
      MPI_Barrier(comm_spec.comm());
    }

    vertex_map.reset();
    vertex_map.fnum_ = fnum;
    vertex_map.is_global_ = is_global_;
    vertex_map.partitioner_ = partitioner_;
    this->partitioner_ = nullptr;
    vertex_map.idxers_.resize(fnum, nullptr);
    for (fid_t fid = 0; fid < fnum; ++fid) {
      vertex_map.idxers_[fid] = idxer_builders_[fid]->finish();
      delete idxer_builders_[fid];
      idxer_builders_[fid] = nullptr;
    }
    idxer_builders_.clear();
    vertex_map.id_parser_.init(fnum);

    vertex_map.inner_vertex_size_.resize(fnum, 0);
    vertex_map.inner_vertex_size_[fid_] = vertex_map.idxers_[fid_]->size();

    sync_comm::AllGather(vertex_map.inner_vertex_size_, comm_spec.comm());

    size_t total = 0;
    for (fid_t i = 0; i < fnum; ++i) {
      total += vertex_map.inner_vertex_size_[i];
    }
    vertex_map.total_vertex_size_ = total;
  }

 private:
  fid_t fid_;
  bool is_global_;
  IPartitioner<OID_T>* partitioner_;
  std::vector<IdxerBuilderBase<OID_T, VID_T>*> idxer_builders_;
};

template <typename OID_T, typename VID_T>
void VertexMap<OID_T, VID_T>::UpdateToBalance(
    const CommSpec& comm_spec, const std::vector<VID_T>& vnum_list,
    const std::vector<std::vector<VID_T>>& gid_maps) {
  fid_t fnum = comm_spec.fnum();
  std::vector<std::vector<oid_t>> oid_lists(fnum);
  std::vector<std::vector<VID_T>> unresolved_lids(fnum);
  std::vector<std::vector<std::pair<fid_t, VID_T>>> unresolved_vertices(fnum);
  std::vector<std::vector<OID_T>> unresolved_oids(fnum);
  for (fid_t fid = 0; fid < fnum; ++fid) {
    VID_T num = inner_vertex_size_[fid];
    CHECK_EQ(num, gid_maps[fid].size());
    for (VID_T lid = 0; lid < num; ++lid) {
      VID_T new_gid = gid_maps[fid][lid];
      internal_oid_t oid;
      fid_t new_fid = GetFidFromGid(new_gid);
      VID_T new_lid = GetLidFromGid(new_gid);
      if (!idxers_[fid]->get_key(lid, oid)) {
        unresolved_lids[fid].push_back(lid);
        unresolved_vertices[fid].push_back(std::make_pair(new_fid, new_lid));
      } else {
        if (oid_lists[new_fid].size() <= new_lid) {
          oid_lists[new_fid].resize(new_lid + 1);
        }
        oid_lists[new_fid][new_lid] = oid_t(oid);
      }
    }
  }
  LOG(INFO) << "after init oid lists";
  {
    std::thread request_thread = std::thread([&]() {
      int src_worker_id = (comm_spec.worker_id() + 1) % comm_spec.worker_num();
      while (src_worker_id != comm_spec.worker_id()) {
        for (fid_t fid = 0; fid < fnum; ++fid) {
          if (comm_spec.FragToWorker(fid) != src_worker_id) {
            continue;
          }
          sync_comm::Send(unresolved_lids[fid], src_worker_id, 0,
                          comm_spec.comm());
          sync_comm::Recv(unresolved_oids[fid], src_worker_id, 1,
                          comm_spec.comm());
        }
        src_worker_id = (src_worker_id + 1) % comm_spec.worker_num();
      }
    });
    std::thread response_thread = std::thread([&]() {
      int dst_worker_id = (comm_spec.worker_id() + comm_spec.worker_num() - 1) %
                          comm_spec.worker_num();
      while (dst_worker_id != comm_spec.worker_id()) {
        for (fid_t fid = 0; fid < fnum; ++fid) {
          if (comm_spec.FragToWorker(fid) != comm_spec.worker_id()) {
            continue;
          }
          std::vector<VID_T> lid_list;
          sync_comm::Recv(lid_list, dst_worker_id, 0, comm_spec.comm());
          std::vector<OID_T> oid_list;
          for (auto lid : lid_list) {
            OID_T oid{};
            if (!GetOid(fid, lid, oid)) {
              LOG(ERROR) << "Cannot find oid for lid " << lid;
            }
            oid_list.push_back(oid);
          }
          sync_comm::Send(oid_list, dst_worker_id, 1, comm_spec.comm());
        }
        dst_worker_id = (dst_worker_id + comm_spec.worker_num() - 1) %
                        comm_spec.worker_num();
      }
    });
    response_thread.join();
    request_thread.join();
    MPI_Barrier(comm_spec.comm());
  }
  LOG(INFO) << "after resolved oids";
  for (fid_t fid = 0; fid < fnum; ++fid) {
    for (size_t i = 0; i < unresolved_lids[fid].size(); ++i) {
      OID_T oid = unresolved_oids[fid][i];
      const auto& pair = unresolved_vertices[fid][i];
      oid_lists[pair.first][pair.second] = oid;
    }
  }
  LOG(INFO) << "after update oid lists";

  MapPartitioner<OID_T>* new_partitioner = new MapPartitioner<OID_T>();
  new_partitioner->Init(oid_lists);

  VertexMapBuilder<OID_T, VID_T> builder(comm_spec.fid(), comm_spec.fnum(),
                                         new_partitioner, true);
  for (auto& oid : oid_lists[comm_spec.fid()]) {
    internal_oid_t internal_oid(oid);
    builder.add_vertex(internal_oid);
  }
  LOG(INFO) << "after add vertices";

  builder.finish(comm_spec, *this);
  LOG(INFO) << "after finish";
}

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_VERTEX_MAP_BETA_H_
