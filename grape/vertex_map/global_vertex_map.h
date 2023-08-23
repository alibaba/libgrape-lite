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
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/vertex_map/vertex_map_base.h"
#include "grape/worker/comm_spec.h"

template <typename Key, typename Value>
using HashMap = ska::flat_hash_map<Key, Value>;

#define SORT_DISTINCT
namespace grape {

/**
 * @brief a kind of VertexMapBase which holds global mapping information in
 * each worker.
 *
 * @tparam OID_T
 * @tparam VID_T
 */
template <typename OID_T, typename VID_T>
class GlobalVertexMap : public VertexMapBase<OID_T, VID_T> {
  // TODO(lxj): to support shared-memory for workers on same host (auto apps)

  using Base = VertexMapBase<OID_T, VID_T>;

 public:
  explicit GlobalVertexMap(const CommSpec& comm_spec) : Base(comm_spec) {}
  ~GlobalVertexMap() = default;
  void Init() {
    Base::Init();
    o2l_.resize(Base::GetCommSpec().fnum());
    l2o_.resize(Base::GetCommSpec().fnum());
  }

  size_t GetTotalVertexSize() {
    size_t size = 0;
    for (const auto& v : o2l_) {
      size += v.size();
    }
    return size;
  }

  size_t GetInnerVertexSize(fid_t fid) { return l2o_[fid].size(); }
  void Clear() {}
  void AddVertex(fid_t fid, const OID_T& oid) {
#ifdef SORT_DISTINCT
    l2o_[fid].emplace_back(oid);
#else
    auto& rm = o2l_[fid];
    if (rm.find(oid) == rm.end()) {
      rm.emplace(oid, static_cast<VID_T>(l2o_[fid].size()));
      l2o_[fid].emplace_back(oid);
    }
#endif
  }

  bool AddVertex(fid_t fid, const OID_T& oid, VID_T& gid) {
    auto& rm = o2l_[fid];
    auto iter = rm.find(oid);
    if (iter == rm.end()) {
      gid = static_cast<VID_T>(l2o_[fid].size());
      rm.emplace(oid, gid);
      l2o_[fid].emplace_back(oid);
      gid = Base::Lid2Gid(fid, gid);
      return true;
    } else {
      gid = Base::Lid2Gid(fid, iter->second);
      return false;
    }
  }

  bool GetOid(const VID_T& gid, OID_T& oid) {
    fid_t fid = Base::GetFidFromGid(gid);
    VID_T lid = Base::GetLidFromGid(gid);
    return GetOid(fid, lid, oid);
  }

  bool GetOid(fid_t fid, const VID_T& lid, OID_T& oid) {
    if (lid >= l2o_[fid].size()) {
      return false;
    }
    oid = l2o_[fid][lid];
    return true;
  }

  bool GetGid(fid_t fid, const OID_T& oid, VID_T& gid) {
    auto& rm = o2l_[fid];
    auto iter = rm.find(oid);
    if (iter == rm.end()) {
      return false;
    } else {
      gid = Base::Lid2Gid(fid, iter->second);
      return true;
    }
  }

  bool GetGid(const OID_T& oid, VID_T& gid) {
    for (fid_t i = 0; i < Base::GetFragmentNum(); ++i) {
      if (GetGid(i, oid, gid)) {
        return true;
      }
    }
    return false;
  }

  void Construct() {
    const CommSpec& comm_spec = Base::GetCommSpec();
    int worker_id = comm_spec.worker_id();
    int worker_num = comm_spec.worker_num();
    bool to_sort = false;
    for (fid_t fid = 0; fid != comm_spec.fnum(); ++fid) {
      if (!l2o_[fid].empty()) {
        to_sort = o2l_[fid].empty();
        break;
      }
    }
    if (to_sort) {
      fid_t fid = comm_spec.fid();
      auto& vec = l2o_[fid];
      auto& rm = o2l_[fid];
      std::sort(vec.begin(), vec.end());
      size_t vnum = vec.size();
      size_t count = 0;
      for (size_t i = 1; i < vnum; ++i) {
        if (vec[i] == vec[i - 1]) {
          ++count;
        } else {
          vec[i - count] = vec[i];
        }
      }
      vec.resize(vnum - count);
      vec.shrink_to_fit();
      VID_T distincted_vnum = static_cast<VID_T>(vec.size());
      rm.reserve(distincted_vnum);
      for (VID_T lid = 0; lid < distincted_vnum; ++lid) {
        rm.emplace(vec[lid], lid);
      }
    }
    {
      std::thread recv_thread([&]() {
        int src_worker_id = (worker_id + 1) % worker_num;
        while (src_worker_id != worker_id) {
          for (fid_t fid = 0; fid < Base::GetFragmentNum(); ++fid) {
            if (comm_spec.FragToWorker(fid) != src_worker_id) {
              continue;
            }
            RecvVector(l2o_[fid], src_worker_id, comm_spec.comm());
          }
          src_worker_id = (src_worker_id + 1) % worker_num;
        }
      });
      std::thread send_thread([&]() {
        int dst_worker_id = (worker_id + worker_num - 1) % worker_num;
        while (dst_worker_id != worker_id) {
          for (fid_t fid = 0; fid < Base::GetFragmentNum(); ++fid) {
            if (comm_spec.FragToWorker(fid) != worker_id) {
              continue;
            }
            SendVector(l2o_[fid], dst_worker_id, comm_spec.comm());
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
      fid_t fnum = comm_spec.fnum();
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
            auto& rm = o2l_[got];
            auto& ol = l2o_[got];
            VID_T vnum = static_cast<VID_T>(ol.size());
            rm.reserve(vnum);
            for (VID_T lid = 0; lid < vnum; ++lid) {
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

  template <typename IOADAPTOR_T>
  void Serialize(const std::string& prefix) {
    char fbuf[1024];
    snprintf(fbuf, sizeof(fbuf), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);

    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(fbuf)));
    io_adaptor->Open("wb");

    InArchive ia;
    Base::BaseSerialize(ia);
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();
    for (fid_t i = 0; i < Base::GetFragmentNum(); ++i) {
      ia << l2o_[i].size();
    }
    CHECK(io_adaptor->WriteArchive(ia));
    ia.Clear();
    for (fid_t i = 0; i < Base::GetFragmentNum(); ++i) {
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

    OutArchive oa;
    CHECK(io_adaptor->ReadArchive(oa));
    Base::BaseDeserialize(oa);
    oa.Clear();

    l2o_.clear();
    l2o_.resize(Base::GetFragmentNum());
    o2l_.clear();
    o2l_.resize(Base::GetFragmentNum());

    CHECK(io_adaptor->ReadArchive(oa));
    for (fid_t i = 0; i < Base::GetFragmentNum(); ++i) {
      size_t size;
      oa >> size;
      l2o_[i].resize(size);
    }
    oa.Clear();

    for (fid_t i = 0; i < Base::GetFragmentNum(); ++i) {
      CHECK(io_adaptor->Read(l2o_[i].data(), l2o_[i].size() * sizeof(OID_T)));
    }
    io_adaptor->Close();

    {
      int thread_num = (std::thread::hardware_concurrency() +
                        Base::GetCommSpec().local_num() - 1) /
                       Base::GetCommSpec().local_num();
      std::vector<std::thread> construct_threads(thread_num);
      std::atomic<fid_t> current_fid(0);
      fid_t fnum = Base::GetFragmentNum();
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
    fid_t fnum = Base::GetFragmentNum();
    for (fid_t fid = 0; fid < fnum; ++fid) {
      new_l2o[fid].clear();
      new_l2o[fid].resize(vnum_list[fid]);
    }
    for (fid_t fid = 0; fid < fnum; ++fid) {
      auto& hmap = o2l_[fid];
      for (auto& pair : hmap) {
        VID_T new_gid = gid_maps[fid][pair.second];
        fid_t new_fid = Base::GetFidFromGid(new_gid);
        VID_T new_lid = Base::GetLidFromGid(new_gid);
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
  std::vector<HashMap<OID_T, VID_T>> o2l_;
  std::vector<std::vector<OID_T>> l2o_;
};

}  // namespace grape

#ifdef SORT_DISTINCT
#undef SORT_DISTINCT
#endif  // SORT_DISTINCT

#endif  // GRAPE_VERTEX_MAP_GLOBAL_VERTEX_MAP_H_
