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
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "flat_hash_map/flat_hash_map.hpp"
#include "grape/config.h"
#include "grape/fragment/partitioner.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/vertex_map/vertex_map_base.h"
#include "grape/worker/comm_spec.h"

template <typename Key, typename Value>
using HashMap = ska::flat_hash_map<Key, Value>;

namespace grape {

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class GlobalVertexMap;

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class GlobalVertexMapBuilder {
 private:
  GlobalVertexMapBuilder(fid_t fid, HashMap<OID_T, VID_T>& hmap,
                         std::vector<OID_T>& list,
                         const PARTITIONER_T& partitioner,
                         const IdParser<VID_T>& id_parser)
      : fid_(fid),
        map_(hmap),
        list_(list),
        partitioner_(partitioner),
        id_parser_(id_parser),
        init_size_(list.size()) {}

 public:
  ~GlobalVertexMapBuilder() {}

  void add_vertex(const OID_T& id) {
    assert(partitioner_.GetPartitionId(id) == fid_);
    if (map_.find(id) == map_.end()) {
      map_.emplace(id, static_cast<VID_T>(list_.size()));
      list_.emplace_back(id);
    }
  }

  bool add_vertex(const OID_T& id, VID_T& gid) {
    assert(partitioner_.GetPartitionId(id) == fid_);
    auto iter = map_.find(id);
    if (iter == map_.end()) {
      gid = static_cast<VID_T>(list_.size());
      map_.emplace(id, gid);
      list_.emplace_back(id);
      gid = id_parser_.generate_global_id(fid_, gid);
      return true;
    } else {
      gid = id_parser_.generate_global_id(fid_, iter->second);
      return false;
    }
  }

  void finish(GlobalVertexMap<OID_T, VID_T, PARTITIONER_T>& vertex_map) {
    const CommSpec& comm_spec = vertex_map.GetCommSpec();
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    fid_t fnum = comm_spec.fnum();
    std::vector<size_t> init_sizes(fnum);
    {
      std::thread recv_thread([&]() {
        int src_worker_id = (worker_id + 1) % worker_num;
        while (src_worker_id != worker_id) {
          for (fid_t fid = 0; fid < fnum; ++fid) {
            if (comm_spec.FragToWorker(fid) != src_worker_id) {
              continue;
            }
            init_sizes[fid] = vertex_map.l2o_[fid].size();
            sync_comm::RecvAt<OID_T>(vertex_map.l2o_[fid],
                                     vertex_map.l2o_[fid].size(), src_worker_id,
                                     0, comm_spec.comm());
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
            sync_comm::SendPartial<OID_T>(list_, init_size_, list_.size(),
                                          dst_worker_id, 0, comm_spec.comm());
          }
          dst_worker_id = (dst_worker_id + worker_num - 1) % worker_num;
        }
      });
      send_thread.join();
      recv_thread.join();
    }
    {
      int thread_num =
          (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
          comm_spec.local_num();
      std::atomic<fid_t> current_fid(0);
      std::vector<std::thread> work_threads(thread_num);
      for (int tid = 0; tid < thread_num; ++tid) {
        work_threads[tid] = std::thread([&] {
          fid_t got;
          while (true) {
            got = current_fid.fetch_add(1, std::memory_order_relaxed);
            if (got >= fnum) {
              break;
            }
            if (comm_spec.FragToWorker(got) == worker_id) {
              continue;
            }
            auto& rm = vertex_map.o2l_[got];
            auto& ol = vertex_map.l2o_[got];
            VID_T vnum = static_cast<VID_T>(ol.size());
            rm.reserve(vnum);
            for (VID_T lid = init_sizes[got]; lid < vnum; ++lid) {
              rm.emplace(ol[lid], lid);
            }
          }
        });
      }
      for (auto& thrd : work_threads) {
        thrd.join();
      }
    }
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class GlobalVertexMap;

  fid_t fid_;
  HashMap<OID_T, VID_T>& map_;
  std::vector<OID_T>& list_;
  const PARTITIONER_T& partitioner_;
  const IdParser<VID_T> id_parser_;

  VID_T init_size_;
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

 public:
  explicit GlobalVertexMap(const CommSpec& comm_spec) : base_t(comm_spec) {}
  ~GlobalVertexMap() = default;
  void Init() {
    o2l_.resize(comm_spec_.fnum());
    l2o_.resize(comm_spec_.fnum());
  }

  size_t GetTotalVertexSize() const {
    size_t size = 0;
    for (const auto& v : o2l_) {
      size += v.size();
    }
    return size;
  }

  size_t GetInnerVertexSize(fid_t fid) const { return l2o_[fid].size(); }
  void AddVertex(const OID_T& oid) {
    fid_t fid = partitioner_.GetPartitionId(oid);
    auto& rm = o2l_[fid];
    if (rm.find(oid) == rm.end()) {
      rm.emplace(oid, static_cast<VID_T>(l2o_[fid].size()));
      l2o_[fid].emplace_back(oid);
    }
  }

  using base_t::Lid2Gid;
  bool AddVertex(const OID_T& oid, VID_T& gid) {
    fid_t fid = partitioner_.GetPartitionId(oid);
    auto& rm = o2l_[fid];
    auto iter = rm.find(oid);
    if (iter == rm.end()) {
      gid = static_cast<VID_T>(l2o_[fid].size());
      rm.emplace(oid, gid);
      l2o_[fid].emplace_back(oid);
      gid = Lid2Gid(fid, gid);
      return true;
    } else {
      gid = Lid2Gid(fid, iter->second);
      return false;
    }
  }

  using base_t::GetFidFromGid;
  using base_t::GetLidFromGid;
  bool GetOid(const VID_T& gid, OID_T& oid) const {
    fid_t fid = GetFidFromGid(gid);
    VID_T lid = GetLidFromGid(gid);
    return GetOid(fid, lid, oid);
  }

  bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) const {
    if (lid >= l2o_[fid].size()) {
      return false;
    }
    oid = l2o_[fid][lid];
    return true;
  }

  bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) const {
    auto& rm = o2l_[fid];
    auto iter = rm.find(oid);
    if (iter == rm.end()) {
      return false;
    } else {
      gid = Lid2Gid(fid, iter->second);
      return true;
    }
  }

  bool GetGid(const OID_T& oid, VID_T& gid) const {
    fid_t fid = partitioner_.GetPartitionId(oid);
    return GetGid(fid, oid, gid);
  }

  GlobalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T> GetLocalBuilder() {
    fid_t fid = comm_spec_.fid();
    return GlobalVertexMapBuilder<OID_T, VID_T, PARTITIONER_T>(
        fid, o2l_[fid], l2o_[fid], partitioner_, id_parser_);
  }

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    base_t::serialize(io_adaptor);
    InArchive ia;
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      ia << l2o_[i].size();
    }
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      CHECK(io_adaptor->Write(l2o_[i].data(), l2o_[i].size() * sizeof(OID_T)));
    }
    io_adaptor->Close();
  }

  template <typename IOADAPTOR_T>
  void Deserialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open();

    base_t::deserialize(io_adaptor);

    OutArchive oa;

    l2o_.clear();
    l2o_.resize(comm_spec_.fnum());
    o2l_.clear();
    o2l_.resize(comm_spec_.fnum());

    CHECK(io_adaptor->ReadArchive(oa));
    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      size_t size;
      oa >> size;
      l2o_[i].resize(size);
    }
    oa.Clear();

    for (fid_t i = 0; i < comm_spec_.fnum(); ++i) {
      CHECK(io_adaptor->Read(l2o_[i].data(), l2o_[i].size() * sizeof(OID_T)));
    }
    io_adaptor->Close();

    {
      int thread_num =
          (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
          comm_spec_.local_num();
      std::vector<std::thread> construct_threads(thread_num);
      std::atomic<fid_t> current_fid(0);
      fid_t fnum = comm_spec_.fnum();
      for (int i = 0; i < thread_num; ++i) {
        construct_threads[i] = std::thread([&]() {
          fid_t got;
          while (true) {
            got = current_fid.fetch_add(1, std::memory_order_relaxed);
            if (got >= fnum) {
              break;
            }
            auto& rm = o2l_[got];
            auto& vec = l2o_[got];
            size_t vnum = vec.size();
            rm.reserve(vnum);
            for (size_t lid = 0; lid < vnum; ++lid) {
              rm.emplace(vec[lid], static_cast<VID_T>(lid));
            }
          }
        });
      }

      for (auto& thrd : construct_threads) {
        thrd.join();
      }
    }
  }

  void UpdateToBalance(std::vector<VID_T>& vnum_list,
                       std::vector<std::vector<VID_T>>& gid_maps) {
    std::vector<HashMap<OID_T, VID_T>> new_o2l(o2l_.size());
    std::vector<std::vector<OID_T>> new_l2o(l2o_.size());
    fid_t fnum = comm_spec_.fnum();
    for (fid_t fid = 0; fid < fnum; ++fid) {
      new_l2o[fid].clear();
      new_l2o[fid].resize(vnum_list[fid]);
    }
    for (fid_t fid = 0; fid < fnum; ++fid) {
      auto& hmap = o2l_[fid];
      for (auto& pair : hmap) {
        VID_T new_gid = gid_maps[fid][pair.second];
        fid_t new_fid = GetFidFromGid(new_gid);
        if (new_fid != fid) {
          partitioner_.SetPartitionId(pair.first, new_fid);
        }
        VID_T new_lid = GetLidFromGid(new_gid);
        new_l2o[new_fid][new_lid] = pair.first;
        new_o2l[new_fid].emplace(pair.first, new_lid);
      }
      HashMap<OID_T, VID_T> tmp;
      hmap.swap(tmp);
    }
    o2l_.swap(new_o2l);
    l2o_.swap(new_l2o);
  }

 private:
  template <typename _OID_T, typename _VID_T, typename _PARTITIONER_T>
  friend class GlobalVertexMapBuilder;

  std::vector<HashMap<OID_T, VID_T>> o2l_;
  std::vector<std::vector<OID_T>> l2o_;
  using base_t::comm_spec_;
  using base_t::id_parser_;
  using base_t::partitioner_;
};

}  // namespace grape

#endif  // GRAPE_VERTEX_MAP_GLOBAL_VERTEX_MAP_H_
