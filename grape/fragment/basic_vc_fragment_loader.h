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

#ifndef GRAPE_FRAGMENT_BASIC_VC_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_BASIC_VC_FRAGMENT_LOADER_H_

namespace grape {

template <typename FRAG_T>
class BasicVCFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;
  using edge_t = typename fragment_t::edge_t;
  using vertices_t = typename fragment_t::vertices_t;

 public:
  explicit BasicVCFragmentLoader(const CommSpec& comm_spec, int64_t vnum)
      : comm_spec_(comm_spec), partitioner_(comm_spec.fnum(), vnum) {
    comm_spec_.Dup();

    edges_to_frag_.resize(comm_spec_.fnum());
    for (fid_t fid = 0; fid < comm_spec_.fnum(); ++fid) {
      int worker_id = comm_spec_.FragToWorker(fid);
      edges_to_frag_[fid].Init(comm_spec_.comm(), edge_tag, 4096000);
      edges_to_frag_[fid].SetDestination(worker_id, fid);
      if (worker_id == comm_spec_.worker_id()) {
        edges_to_frag_[fid].DisableComm();
      }
    }
    edge_recv_thread_ =
        std::thread(&BasicVCFragmentLoader::edgeRecvRoutine, this);
    recv_thread_running_ = true;
  }

  void AddEdge(const oid_t& src, const oid_t& dst, const edata_t& data) {
    fid_t fid = partitioner_.get_edge_partition(src, dst);
    edges_to_frag_[fid].Emplace(src, dst, data);
  }

  void ConstructFragment(std::shared_ptr<fragment_t>& fragment) {
    for (auto& e : edges_to_frag_) {
      e.Flush();
    }
    edge_recv_thread_.join();
    recv_thread_running_ = false;

    MPI_Barrier(comm_spec_.comm());
    got_edges_.emplace_back(
        std::move(edges_to_frag_[comm_spec_.fid()].buffers()));
    edges_to_frag_[comm_spec_.fid()].Clear();

    std::vector<edge_t> edges;
    size_t edge_num = 0;
    for (auto& buffer : got_edges_) {
      edge_num += buffer.size();
    }
    edges.reserve(edge_num);
    for (auto& buffer : got_edges_) {
      foreach_rval(buffer, [&edges](oid_t&& src, oid_t&& dst, edata_t&& data) {
        edges.emplace_back(src, dst, data);
      });
    }

    fragment.reset(new fragment_t());
    fragment->Init(comm_spec_, vnum_, std::move(edges));
  }

 private:
  void edgeRecvRoutine() {
    ShuffleIn<oid_t, oid_t, edata_t> data_in;
    data_in.Init(comm_spec_.fnum(), comm_spec_.comm(), edge_tag);
    fid_t dst_fid;
    int src_worker_id;
    while (!data_in.Finished()) {
      src_worker_id = data_in.Recv(dst_fid);
      if (src_worker_id == -1) {
        break;
      }
      CHECK_EQ(dst_fid, comm_spec_.fid());
      got_edges_.emplace_back(std::move(data_in.buffers()));
      data_in.Clear();
    }
  }

  CommSpec comm_spec_;
  int64_t vnum_;
  VCPartitioner<int64_t> partitioner_;

  std::vector<ShuffleBufferTuple<oid_t, oid_t, edata_t>> got_edges_;

  std::vector<ShuffleOut<oid_t, oid_t, edata_t>> edges_to_frag_;
  std::thread edge_recv_thread_;
  bool recv_thread_running_;

  static constexpr int edge_tag = 6;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_BASIC_VC_FRAGMENT_LOADER_H_
