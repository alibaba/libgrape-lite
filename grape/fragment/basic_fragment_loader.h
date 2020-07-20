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

#ifndef GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <fstream>

#include "grape/communication/shuffle.h"
#include "grape/config.h"
#include "grape/fragment/rebalancer.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/utils/vertex_array.h"
#include "grape/utils/concurrent_queue.h"
#include "grape/worker/comm_spec.h"

namespace grape {
struct EmptyType;

template <typename OID_T, typename VDATA_T>
inline bool compare_by_id(const internal::Vertex<OID_T, VDATA_T>& lhs,
                          const internal::Vertex<OID_T, VDATA_T>& rhs) {
  return lhs.vid() < rhs.vid();
}

template <typename FRAG_T, typename PARTITIONER_T, typename IOADAPTOR_T,
          typename Enable = void>
class BasicFragmentLoader;

/**
 * @brief BasicFragmentLoader manages temporal vertics/edges added by a more
 * specific loader, shuffles data and builds fragments.
 *
 * @tparam FRAG_T
 * @tparam PARTITIONER_T
 */
template <typename FRAG_T, typename PARTITIONER_T, typename IOADAPTOR_T>
class BasicFragmentLoader<
    FRAG_T, PARTITIONER_T, IOADAPTOR_T,
    typename std::enable_if<
        !std::is_same<typename FRAG_T::vdata_t, EmptyType>::value>::type> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using partitioner_t = PARTITIONER_T;
  using vertex_map_t = typename fragment_t::vertex_map_t;

  static constexpr LoadStrategy load_strategy = fragment_t::load_strategy;

 public:
  explicit BasicFragmentLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec) {
    comm_spec_.Dup();
    vm_ptr_ = std::shared_ptr<vertex_map_t>(new vertex_map_t(comm_spec_));
    vertices_to_frag_.resize(comm_spec_.fnum());
    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      vertices_to_frag_[fid].Init(comm_spec_.comm(), vertex_tag);
      vertices_to_frag_[fid].SetDestination(worker_id, fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        vertices_to_frag_[fid].DisableComm();
        edges_to_frag_[fid].DisableComm();
      }
    }

    recv_thread_running_ = false;
  }

  ~BasicFragmentLoader() { Stop(); }

  void SetPartitioner(const PARTITIONER_T& partitioner) {
    partitioner_ = partitioner;
  }

  void SetPartitioner(PARTITIONER_T&& partitioner) {
    partitioner_ = std::move(partitioner);
  }

  void SetRebalance(bool rebalance, int rebalance_vertex_factor) {
    rebalance_ = rebalance;
    rebalance_vertex_factor_ = rebalance_vertex_factor;
  }

  void Start() {
    vertex_recv_thread_ =
        std::thread(&BasicFragmentLoader::vertexRecvRoutine, this);
    edge_recv_thread_ =
        std::thread(&BasicFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;
  }

  void Stop() {
    if (recv_thread_running_) {
      for (auto& va : vertices_to_frag_) {
        va.Flush();
      }
      for (auto& ea : edges_to_frag_) {
        ea.Flush();
      }
      vertex_recv_thread_.join();
      edge_recv_thread_.join();
      recv_thread_running_ = false;
    }
  }

  void AddVertex(const oid_t& id, const vdata_t& data) {
    fid_t fid = partitioner_.GetPartitionId(id);
    vdata_t ref_data(data);
    vertices_to_frag_[fid].Emplace(id, ref_data);
  }

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t src_fid = partitioner_.GetPartitionId(src);
    fid_t dst_fid = partitioner_.GetPartitionId(dst);
    edata_t ref_data(data);
    edges_to_frag_[src_fid].Emplace(src, dst, ref_data);
    if (src_fid != dst_fid) {
      edges_to_frag_[dst_fid].Emplace(src, dst, ref_data);
    }
  }

  bool SerializeFragment(std::shared_ptr<fragment_t>& fragment,
                         const std::string& serialization_prefix) {
    if (comm_spec_.worker_id() == 0) {
      vm_ptr_->Serialize(serialization_prefix);
    }

    MPI_Barrier(comm_spec_.comm());

    // If not using a nfs, each worker should serialize a copy of vertex map.
    auto exists_file = [](const std::string& name) {
      std::ifstream f(name.c_str());
      return f.good();
    };
    char serial_file[1024];
    snprintf(serial_file, sizeof(serial_file), "%s/%s",
             serialization_prefix.c_str(), kSerializationVertexMapFilename);
    if (comm_spec_.local_id() == 0 && !exists_file(serial_file)) {
      vm_ptr_->Serialize(serialization_prefix);
    }

    fragment->Serialize(serialization_prefix);

    return true;
  }

  bool DeserializeFragment(std::shared_ptr<fragment_t>& fragment,
                           const std::string& deserialization_prefix) {
    auto io_adaptor =
        std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(deserialization_prefix));
    if (io_adaptor->IsExist()) {
      vm_ptr_->Deserialize(deserialization_prefix);
      fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr_));
      fragment->Deserialize(deserialization_prefix, comm_spec_.fid());
    }
    return true;
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) {
    for (auto& va : vertices_to_frag_) {
      va.Flush();
    }
    for (auto& ea : edges_to_frag_) {
      ea.Flush();
    }
    vertex_recv_thread_.join();
    edge_recv_thread_.join();
    recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());

    got_vertices_id_.emplace_back(
        std::move(vertices_to_frag_[comm_spec_.fid()].Buffer0()));
    got_vertices_data_.emplace_back(std::move(std::vector<vdata_t>(
        std::move(vertices_to_frag_[comm_spec_.fid()].Buffer1()))));
    got_edges_src_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].Buffer0()));
    got_edges_dst_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].Buffer1()));
    got_edges_data_.emplace_back(std::move(std::vector<edata_t>(
        std::move(edges_to_frag_[comm_spec_.fid()].Buffer2()))));

    sortDistinct();

    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "]: finished construct vertex map and process vertices";

    processEdges();

    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "]: finished process edges";

    fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr_));
    fragment->Init(comm_spec_.fid(), processed_vertices_, processed_edges_);

    initMirrorInfo(fragment);
    initOuterVertexData(fragment);
  }

 private:
  void processEdgesRoutine(std::vector<std::vector<oid_t>>& edge_src,
                           std::vector<std::vector<oid_t>>& edge_dst,
                           std::vector<std::vector<edata_t>>& edge_data,
                           std::vector<Edge<vid_t, edata_t>>& to) {
    to.clear();
    size_t edge_num = 0;
    for (auto& ed : edge_data) {
      edge_num += ed.size();
    }
    to.resize(edge_num);
    fid_t src_fid, dst_fid;
    vid_t src_gid, dst_gid;
    auto to_iter = to.begin();
    size_t buf_num = edge_data.size();
    for (size_t buf_id = 0; buf_id < buf_num; ++buf_id) {
      auto src_iter = edge_src[buf_id].begin();
      auto src_end = edge_src[buf_id].end();
      auto dst_iter = edge_dst[buf_id].begin();
      auto data_iter = edge_data[buf_id].begin();
      while (src_iter != src_end) {
        src_fid = partitioner_.GetPartitionId(*src_iter);
        vm_ptr_->GetGid(src_fid, oid_t(*src_iter), src_gid);
        dst_fid = partitioner_.GetPartitionId(*dst_iter);
        vm_ptr_->GetGid(dst_fid, oid_t(*dst_iter), dst_gid);
        to_iter->set_edata(std::move(*data_iter));
        to_iter->SetEndpoint(src_gid, dst_gid);
        ++src_iter;
        ++dst_iter;
        ++data_iter;
        ++to_iter;
      }
    }
  }

  void processEdges() {
    processEdgesRoutine(got_edges_src_, got_edges_dst_, got_edges_data_,
                        processed_edges_);
    got_edges_src_.clear();
    got_edges_dst_.clear();
    got_edges_data_.clear();
  }

  void sortThreadRoutine(
      fid_t fid, std::vector<std::vector<oid_t>>& vertex_id,
      std::vector<std::vector<vdata_t>>& vertex_data,
      std::vector<internal::Vertex<vid_t, vdata_t>>& vertices) {
    vertices.clear();
    size_t buf_num = vertex_id.size();
    for (size_t buf_id = 0; buf_id < buf_num; ++buf_id) {
      auto& id_list = vertex_id[buf_id];
      auto& data_list = vertex_data[buf_id];
      size_t index = 0;
      vid_t gid;
      for (auto& id : id_list) {
        if (vm_ptr_->AddVertex(fid, id, gid)) {
          vertices.emplace_back(gid, data_list[index]);
        }
        ++index;
      }
    }
  }

  void sortDistinct() {
    vm_ptr_->Init();
    sortThreadRoutine(comm_spec_.fid(), got_vertices_id_, got_vertices_data_,
                      processed_vertices_);
    got_vertices_id_.clear();
    got_vertices_data_.clear();
    vm_ptr_->Construct();
  }

  void vertexRecvRoutine() {
    ShuffleInPair<oid_t, vdata_t> data_in(comm_spec_.fnum() - 1);
    data_in.Init(comm_spec_.comm(), vertex_tag);
    fid_t dst_fid;
    int src_worker_id;
    while (!data_in.Finished()) {
      src_worker_id = data_in.Recv(dst_fid);
      if (src_worker_id == -1) {
        break;
      }
      auto& dst_buf0 = got_vertices_id_;
      auto& dst_buf1 = got_vertices_data_;
      dst_buf0.emplace_back(std::move(data_in.Buffer0()));
      dst_buf1.emplace_back(std::move(data_in.Buffer1()));
    }
  }

  void edgeRecvRoutine() {
    ShuffleInTriple<oid_t, oid_t, edata_t> data_in(comm_spec_.fnum() - 1);
    data_in.Init(comm_spec_.comm(), edge_tag);
    fid_t dst_fid;
    int src_worker_id;
    while (!data_in.Finished()) {
      src_worker_id = data_in.Recv(dst_fid);
      if (src_worker_id == -1) {
        break;
      }
      CHECK_EQ(dst_fid, comm_spec_.fid());
      got_edges_src_.emplace_back(std::move(data_in.Buffer0()));
      got_edges_dst_.emplace_back(std::move(data_in.Buffer1()));
      got_edges_data_.emplace_back(std::move(data_in.Buffer2()));
    }
  }

  void initMirrorInfo(std::shared_ptr<fragment_t> fragment) {
    int worker_id = comm_spec_.worker_id();
    int worker_num = comm_spec_.worker_num();

    std::thread send_thread([&]() {
      std::vector<vid_t> gid_list;
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker_id = (worker_id + i) % worker_num;
        fid_t dst_fid = comm_spec_.WorkerToFrag(dst_worker_id);
        auto range = fragment->OuterVertices(dst_fid);
        vid_t offsets[2];
        offsets[0] = range.begin().GetValue();
        offsets[1] = range.end().GetValue();
        MPI_Send(&offsets[0], sizeof(vid_t) * 2, MPI_CHAR, dst_worker_id, 0,
                 comm_spec_.comm());
        gid_list.clear();
        gid_list.reserve(range.size());
        for (auto v : range) {
          gid_list.push_back(fragment->Vertex2Gid(v));
        }
        MPI_Send(&gid_list[0], sizeof(vid_t) * gid_list.size(), MPI_CHAR,
                 dst_worker_id, 0, comm_spec_.comm());
      }
    });

    std::thread recv_thread([&]() {
      std::vector<vid_t> gid_list;
      for (int i = 1; i < worker_num; ++i) {
        int src_worker_id = (worker_id + worker_num - i) % worker_num;
        fid_t src_fid = comm_spec_.WorkerToFrag(src_worker_id);
        vid_t offsets[2];
        MPI_Recv(&offsets[0], sizeof(vid_t) * 2, MPI_CHAR, src_worker_id, 0,
                 comm_spec_.comm(), MPI_STATUS_IGNORE);
        VertexRange<vid_t> range(offsets[0], offsets[1]);
        gid_list.clear();
        gid_list.resize(range.size());
        MPI_Recv(&gid_list[0], gid_list.size() * sizeof(vid_t), MPI_CHAR,
                 src_worker_id, 0, comm_spec_.comm(), MPI_STATUS_IGNORE);
        fragment->SetupMirrorInfo(src_fid, range, gid_list);
      }
    });

    recv_thread.join();
    send_thread.join();
  }

  void initOuterVertexData(std::shared_ptr<fragment_t> fragment) {
    int worker_id = comm_spec_.worker_id();
    int worker_num = comm_spec_.worker_num();

    std::thread send_thread([&]() {
      InArchive arc;
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker_id = (worker_id + i) % worker_num;
        fid_t dst_fid = comm_spec_.WorkerToFrag(dst_worker_id);
        arc.Clear();
        auto& vertices = fragment->MirrorVertices(dst_fid);
        for (auto v : vertices) {
          arc << fragment->GetData(v);
        }
        MPI_Send(arc.GetBuffer(), arc.GetSize(), MPI_CHAR, dst_worker_id, 0,
                 comm_spec_.comm());
      }
    });

    std::thread recv_thread([&]() {
      OutArchive arc;
      for (int i = 1; i < worker_num; ++i) {
        int src_worker_id = (worker_id + worker_num - i) % worker_num;
        fid_t src_fid = comm_spec_.WorkerToFrag(src_worker_id);
        MPI_Status status;
        MPI_Probe(src_worker_id, 0, comm_spec_.comm(), &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        arc.Clear();
        arc.Allocate(count);
        MPI_Recv(arc.GetBuffer(), arc.GetSize(), MPI_CHAR, src_worker_id, 0,
                 comm_spec_.comm(), MPI_STATUS_IGNORE);
        auto range = fragment->OuterVertices(src_fid);
        for (auto v : range) {
          vdata_t val;
          arc >> val;
          fragment->SetData(v, val);
        }
      }
    });

    recv_thread.join();
    send_thread.join();
  }

 private:
  CommSpec comm_spec_;
  std::shared_ptr<vertex_map_t> vm_ptr_;

  std::vector<ShuffleOutPair<oid_t, vdata_t>> vertices_to_frag_;
  std::vector<ShuffleOutTriple<oid_t, oid_t, edata_t>> edges_to_frag_;

  std::thread vertex_recv_thread_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  std::vector<std::vector<oid_t>> got_vertices_id_;
  std::vector<std::vector<vdata_t>> got_vertices_data_;

  std::vector<std::vector<oid_t>> got_edges_src_;
  std::vector<std::vector<oid_t>> got_edges_dst_;
  std::vector<std::vector<edata_t>> got_edges_data_;

  std::vector<internal::Vertex<vid_t, vdata_t>> processed_vertices_;
  std::vector<Edge<vid_t, edata_t>> processed_edges_;

  static constexpr int vertex_tag = 5;
  static constexpr int edge_tag = 6;

  PARTITIONER_T partitioner_;

  bool rebalance_;
  int rebalance_vertex_factor_;
};

/**
 * @brief BasicFragmentLoader manages temporal vertics/edges added by a more
 * specific loader, shuffles data and builds fragments.
 *
 * This is a partial specialization for VDATA is EmptyType.
 *
 * @tparam OID_T
 * @tparam vid_t
 * @tparam edata_t
 * @tparam PARTITIONER_T
 */
template <typename FRAG_T, typename PARTITIONER_T, typename IOADAPTOR_T>
class BasicFragmentLoader<
    FRAG_T, PARTITIONER_T, IOADAPTOR_T,
    typename std::enable_if<
        std::is_same<typename FRAG_T::vdata_t, EmptyType>::value>::type> {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using partitioner_t = PARTITIONER_T;
  using vertex_map_t = typename fragment_t::vertex_map_t;

  static constexpr LoadStrategy load_strategy = fragment_t::load_strategy;

 public:
  explicit BasicFragmentLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec) {
    comm_spec_.Dup();
    vm_ptr_ = std::shared_ptr<vertex_map_t>(new vertex_map_t(comm_spec_));
    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag_[fid].DisableComm();
      }
    }

    recv_thread_running_ = false;
  }

  ~BasicFragmentLoader() { Stop(); }

  void SetPartitioner(const PARTITIONER_T& partitioner) {
    partitioner_ = partitioner;
  }

  void SetPartitioner(PARTITIONER_T&& partitioner) {
    partitioner_ = std::move(partitioner);
  }

  void SetRebalance(bool rebalance, int rebalance_vertex_factor) {
    rebalance_ = rebalance;
    rebalance_vertex_factor_ = rebalance_vertex_factor;
  }

  void Start() {
    got_edges_queues_.SetProducerNum(2);

    edge_recv_thread_ =
        std::thread(&BasicFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;

    vm_ptr_->Init();
    construct_vm_thread_ =
        std::thread(&BasicFragmentLoader::constructVMThreadRoutine, this,
                    comm_spec_.fid(), std::ref(got_edges_queues_));
  }

  void Stop() {
    if (recv_thread_running_) {
      for (auto& ea : edges_to_frag_) {
        ea.Flush();
      }
      edge_recv_thread_.join();
      recv_thread_running_ = false;

      got_edges_queues_.DecProducerNum();

      construct_vm_thread_.join();
    }
  }

  void AddVertex(const oid_t& id, const EmptyType& data) {}

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t src_fid = partitioner_.GetPartitionId(src);
    fid_t dst_fid = partitioner_.GetPartitionId(dst);

    edata_t ref_data(data);
    edges_to_frag_[src_fid].Emplace(src, dst, ref_data);
    if (src_fid != dst_fid) {
      edges_to_frag_[dst_fid].Emplace(src, dst, ref_data);
    }
  }

  bool SerializeFragment(std::shared_ptr<fragment_t>& fragment,
                         const std::string prefix) {
    if (comm_spec_.worker_id() == 0) {
      vm_ptr_->template Serialize<IOADAPTOR_T>(prefix);
    }

    MPI_Barrier(comm_spec_.comm());

    // If not using a nfs, each worker should serialize a copy of vertex map.
    auto exists_file = [](const std::string& name) {
      std::ifstream f(name.c_str());
      return f.good();
    };
    char serial_file[1024];
    snprintf(serial_file, sizeof(serial_file), "%s/%s", prefix.c_str(),
             kSerializationVertexMapFilename);
    if (comm_spec_.local_id() == 0 && !exists_file(serial_file)) {
      vm_ptr_->template Serialize<IOADAPTOR_T>(prefix);
    }
    fragment->template Serialize<IOADAPTOR_T>(prefix);
    return true;
  }

  bool DeserializeFragment(std::shared_ptr<fragment_t>& fragment,
                           const std::string prefix) {
    auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(prefix));
    if (io_adaptor->IsExist()) {
      vm_ptr_->template Deserialize<IOADAPTOR_T>(prefix);
      fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr_));
      fragment->template Deserialize<IOADAPTOR_T>(prefix, comm_spec_.fid());
      return true;
    }
    return false;
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) {
    for (auto& ea : edges_to_frag_) {
      ea.Flush();
    }
    edge_recv_thread_.join();
    recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());

    std::tuple<std::vector<oid_t>, std::vector<oid_t>, std::vector<edata_t>>
        item(std::move(edges_to_frag_[comm_spec_.fid()].Buffer0()),
             std::move(edges_to_frag_[comm_spec_.fid()].Buffer1()),
             std::move(edges_to_frag_[comm_spec_.fid()].Buffer2()));
    got_edges_queues_.Put(std::move(item));

    got_edges_queues_.DecProducerNum();

    construct_vm_thread_.join();

    vm_ptr_->Construct();

    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "]: finished construct vertex map and process vertices";

    processEdges();

    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "]: finished process edges";

    if (rebalance_) {
      std::vector<typename fragment_t::vertex_t> fake_vertices;
      Rebalancer<fragment_t> rb(comm_spec_, rebalance_vertex_factor_);
      rb.Rebalance(vm_ptr_, fake_vertices, processed_edges_);
      VLOG(1) << "[worker-" << comm_spec_.worker_id()
              << "]: finished rebalancing";
    }

    fragment = std::shared_ptr<fragment_t>(new fragment_t(vm_ptr_));
    std::vector<internal::Vertex<vid_t, EmptyType>> fake_vertices;
    fragment->Init(comm_spec_.fid(), fake_vertices, processed_edges_);
    VLOG(1) << "[worker-" << comm_spec_.worker_id()
            << "]: finished construction";

    initMirrorInfo(fragment);
  }

 private:
  struct work_unit {
    work_unit(fid_t fid_, size_t index_, size_t begin_)
        : fid(fid_), index(index_), begin(begin_) {}
    fid_t fid;
    size_t index;
    size_t begin;
  };

  void processEdgesRoutine(std::vector<std::vector<oid_t>>& edge_src,
                           std::vector<std::vector<oid_t>>& edge_dst,
                           std::vector<std::vector<edata_t>>& edge_data,
                           std::vector<Edge<vid_t, edata_t>>& to);

  void processEdges() {
    std::vector<work_unit> work_units;
    {
      size_t cur = 0;
      size_t index = 0;
      for (auto& buf : got_edges_src_) {
        work_units.emplace_back(comm_spec_.fid(), index, cur);
        cur += buf.size();
        ++index;
      }
      processed_edges_.resize(cur);
    }
    {
      std::atomic<size_t> current_work_unit(0);
      int thread_num =
          (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
          comm_spec_.local_num();
      std::vector<std::thread> process_threads(thread_num);
      for (int tid = 0; tid < thread_num; ++tid) {
        process_threads[tid] = std::thread([&]() {
          fid_t u_fid, v_fid;
          size_t got;
          while (true) {
            got = current_work_unit.fetch_add(1, std::memory_order_release);
            if (got >= work_units.size()) {
              break;
            }
            auto& wu = work_units[got];
            auto& src_buf = got_edges_src_[wu.index];
            auto& dst_buf = got_edges_dst_[wu.index];
            auto& data_buf = got_edges_data_[wu.index];
            Edge<vid_t, edata_t>* ptr = &processed_edges_[wu.begin];

            auto src_iter = src_buf.begin();
            auto dst_iter = dst_buf.begin();
            auto data_iter = data_buf.begin();
            auto src_end = src_buf.end();

            while (src_iter != src_end) {
              u_fid = partitioner_.GetPartitionId(*src_iter);
              v_fid = partitioner_.GetPartitionId(*dst_iter);
              vm_ptr_->GetGid(u_fid, *src_iter, ptr->src_);
              vm_ptr_->GetGid(v_fid, *dst_iter, ptr->dst_);
              ptr->edata_ = std::move(*data_iter);
              ++src_iter;
              ++dst_iter;
              ++data_iter;
              ++ptr;
            }
          }
        });
      }
      for (auto& thrd : process_threads) {
        thrd.join();
      }
    }
    got_edges_src_.clear();
    got_edges_dst_.clear();
    got_edges_data_.clear();
  }

  void constructVMThreadRoutine(
      fid_t fid,
      BlockingQueue<std::tuple<std::vector<oid_t>, std::vector<oid_t>,
                               std::vector<edata_t>>>& queue) {
    std::tuple<std::vector<oid_t>, std::vector<oid_t>, std::vector<edata_t>>
        in_tuple;

    while (queue.Get(in_tuple)) {
      auto& src_id = std::get<0>(in_tuple);
      auto& dst_id = std::get<1>(in_tuple);
      auto& edge_data = std::get<2>(in_tuple);

      for (auto& id : src_id) {
        fid_t frag_id = partitioner_.GetPartitionId(id);
        if (frag_id == fid) {
          vm_ptr_->AddVertex(fid, id);
        }
      }
      for (auto& id : dst_id) {
        fid_t frag_id = partitioner_.GetPartitionId(id);
        if (frag_id == fid) {
          vm_ptr_->AddVertex(fid, id);
        }
      }

      got_edges_src_.emplace_back(std::move(src_id));
      got_edges_dst_.emplace_back(std::move(dst_id));
      got_edges_data_.emplace_back(
          std::move(std::vector<edata_t>(std::move(edge_data))));
    }
  }

  void edgeRecvRoutine() {
    ShuffleInTriple<oid_t, oid_t, edata_t> data_in(comm_spec_.fnum() - 1);
    data_in.Init(comm_spec_.comm(), edge_tag);
    fid_t dst_fid;
    int src_worker_id;
    while (!data_in.Finished()) {
      src_worker_id = data_in.Recv(dst_fid);
      if (src_worker_id == -1) {
        break;
      }
      CHECK_EQ(dst_fid, comm_spec_.fid());
      std::tuple<std::vector<oid_t>, std::vector<oid_t>, std::vector<edata_t>>
          item(std::move(data_in.Buffer0()), std::move(data_in.Buffer1()),
               std::move(data_in.Buffer2()));
      got_edges_queues_.Put(std::move(item));
    }

    got_edges_queues_.DecProducerNum();
  }

  void initMirrorInfo(std::shared_ptr<fragment_t> fragment) {
    int worker_id = comm_spec_.worker_id();
    int worker_num = comm_spec_.worker_num();

    std::thread send_thread([&]() {
      std::vector<vid_t> gid_list;
      for (int i = 1; i < worker_num; ++i) {
        int dst_worker_id = (worker_id + i) % worker_num;
        fid_t dst_fid = comm_spec_.WorkerToFrag(dst_worker_id);
        auto range = fragment->OuterVertices(dst_fid);
        vid_t offsets[2];
        offsets[0] = range.begin().GetValue();
        offsets[1] = range.end().GetValue();
        MPI_Send(&offsets[0], sizeof(vid_t) * 2, MPI_CHAR, dst_worker_id, 0,
                 comm_spec_.comm());
        gid_list.clear();
        gid_list.reserve(range.size());
        for (auto v : range) {
          gid_list.push_back(fragment->Vertex2Gid(v));
        }
        MPI_Send(&gid_list[0], sizeof(vid_t) * gid_list.size(), MPI_CHAR,
                 dst_worker_id, 0, comm_spec_.comm());
      }
    });

    std::thread recv_thread([&]() {
      std::vector<vid_t> gid_list;
      for (int i = 1; i < worker_num; ++i) {
        int src_worker_id = (worker_id + worker_num - i) % worker_num;
        fid_t src_fid = comm_spec_.WorkerToFrag(src_worker_id);
        vid_t offsets[2];
        MPI_Recv(&offsets[0], sizeof(vid_t) * 2, MPI_CHAR, src_worker_id, 0,
                 comm_spec_.comm(), MPI_STATUS_IGNORE);
        VertexRange<vid_t> range(offsets[0], offsets[1]);
        gid_list.clear();
        gid_list.resize(range.size());
        MPI_Recv(&gid_list[0], gid_list.size() * sizeof(vid_t), MPI_CHAR,
                 src_worker_id, 0, comm_spec_.comm(), MPI_STATUS_IGNORE);
        fragment->SetupMirrorInfo(src_fid, range, gid_list);
      }
    });

    recv_thread.join();
    send_thread.join();
  }

 private:
  CommSpec comm_spec_;
  std::shared_ptr<vertex_map_t> vm_ptr_;

  std::vector<ShuffleOutTriple<oid_t, oid_t, edata_t>> edges_to_frag_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  BlockingQueue<
      std::tuple<std::vector<oid_t>, std::vector<oid_t>, std::vector<edata_t>>>
      got_edges_queues_;
  std::thread construct_vm_thread_;

  std::vector<std::vector<oid_t>> got_edges_src_;
  std::vector<std::vector<oid_t>> got_edges_dst_;
  std::vector<std::vector<edata_t>> got_edges_data_;

  std::vector<Edge<vid_t, edata_t>> processed_edges_;

  static constexpr int edge_tag = 6;

  PARTITIONER_T partitioner_;

  bool rebalance_;
  int rebalance_vertex_factor_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_FRAGMENT_LOADER_H_
