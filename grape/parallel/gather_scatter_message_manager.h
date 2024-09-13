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

#ifndef GRAPE_PARALLEL_GATHER_SCATTER_MESSAGE_MANAGER_H_
#define GRAPE_PARALLEL_GATHER_SCATTER_MESSAGE_MANAGER_H_

namespace grape {

class GatherScatterMessageManager : public MessageManagerBase {
 public:
  GatherScatterMessageManager() : comm_(NULL_COMM) {}
  ~GatherScatterMessageManager() override {
    if (ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  void Init(MPI_Comm comm) override {
    MPI_Comm_dup(comm, &comm_);
    comm_spec_.Init(comm_);
    fid_ = comm_spec_.fid();
    fnum_ = comm_spec_.fnum();

    force_terminate_ = false;
    terminate_info_.Init(fnum_);

    round_ = 0;

    sent_size_ = 0;
  }

  void Start() override {}

  void StartARound() override {}

  void FinishARound() override {}

  bool ToTerminate() override { return force_terminate_; }

  void Finalize() override {
    LOG(INFO) << "[frag-" << fid_ << "] gather comm: " << t0_gather_comm_
              << ", gather calc: " << t1_gather_calc_;
  }

  void ForceContinue() override {}

  void ForceTerminate(const std::string& terminate_info = "") override {
    force_terminate_ = true;
    terminate_info_.info[comm_spec_.fid()] = terminate_info;
  }

  const TerminateInfo& GetTerminateInfo() const override {
    return terminate_info_;
  }

  size_t GetMsgSize() const override { return sent_size_; }

  template <typename GRAPH_T, typename MESSAGE_T, typename AGGR_T>
  void GatherMasterVertices(
      const GRAPH_T& frag,
      const typename GRAPH_T::template both_vertex_array_t<MESSAGE_T>& input,
      typename GRAPH_T::template vertex_array_t<MESSAGE_T>& output) {
    if (!std::is_pod<MESSAGE_T>::value) {
      LOG(FATAL) << "not implemented for non-POD type";
    }

    t0_gather_comm_ -= GetCurrentTime();
    const auto& partitioner = frag.GetPartitioner();
    const auto& input_range = input.GetVertexRange();
    std::vector<MPI_Request> requests;
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (fid_ + i) % fnum_;
      auto target_vertices = partitioner.get_master_vertices(dst_fid);
      if (target_vertices.IsSubsetOf(input_range)) {
        const MESSAGE_T* input_data = &input[*target_vertices.begin()];
        size_t input_size = target_vertices.size();
        sync_comm::isend_buffer(input_data, input_size, dst_fid, 0, comm_,
                                requests);
        sent_size_ += input_size * sizeof(MESSAGE_T);
      } else {
        CHECK(!input_range.OverlapWith(target_vertices))
            << "input: " << input_range.to_string()
            << ", target: " << target_vertices.to_string();
      }
    }

    std::vector<std::vector<MESSAGE_T>> recv_buffers;
    const auto& output_range = output.GetVertexRange();
    size_t output_size = output_range.size();
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto src_vertices = partitioner.get_src_vertices(src_fid);
      if (output_range.IsSubsetOf(src_vertices)) {
        recv_buffers.emplace_back(output_size);
        MemoryInspector::GetInstance().allocate(output_size *
                                                sizeof(MESSAGE_T));
        sync_comm::irecv_buffer(recv_buffers.back().data(), output_size,
                                src_fid, 0, comm_, requests);
        continue;
      } else {
        CHECK(!output_range.OverlapWith(src_vertices));
      }
      auto dst_vertices = partitioner.get_dst_vertices(src_fid);
      if (output_range.IsSubsetOf(dst_vertices)) {
        recv_buffers.emplace_back(output_size);
        MemoryInspector::GetInstance().allocate(output_size *
                                                sizeof(MESSAGE_T));
        sync_comm::irecv_buffer(recv_buffers.back().data(), output_size,
                                src_fid, 0, comm_, requests);
        continue;
      } else {
        CHECK(!output_range.OverlapWith(dst_vertices));
      }
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    t0_gather_comm_ += GetCurrentTime();

    t1_gather_calc_ -= GetCurrentTime();
    MESSAGE_T* output_data = &output[*output_range.begin()];
    std::vector<std::thread> threads;
    int thread_num = std::thread::hardware_concurrency();
    size_t chunk_size =
        std::min<size_t>((output_size + thread_num - 1) / thread_num, 4096);
    std::atomic<size_t> offset(0);
    for (int i = 0; i < thread_num; ++i) {
      threads.emplace_back([&]() {
        if (output_range.IsSubsetOf(input_range)) {
          const MESSAGE_T* to_self = &input[*output_range.begin()];
          while (true) {
            size_t begin = std::min(offset.fetch_add(chunk_size), output_size);
            size_t end = std::min(begin + chunk_size, output_size);
            if (begin == end) {
              break;
            }
            while (begin < end) {
              output_data[begin] = AGGR_T::init();
              AGGR_T::aggregate(output_data[begin], to_self[begin]);
              for (auto& recv_buffer : recv_buffers) {
                AGGR_T::aggregate(output_data[begin], recv_buffer[begin]);
              }
              ++begin;
            }
          }
        } else {
          CHECK(!output_range.OverlapWith(input_range));
          while (true) {
            size_t begin = std::min(offset.fetch_add(chunk_size), output_size);
            size_t end = std::min(begin + chunk_size, output_size);
            if (begin == end) {
              break;
            }
            while (begin < end) {
              output_data[begin] = AGGR_T::init();
              for (auto& recv_buffer : recv_buffers) {
                AGGR_T::aggregate(output_data[begin], recv_buffer[begin]);
              }
              ++begin;
            }
          }
        }
      });
    }
    for (auto& thrd : threads) {
      thrd.join();
    }

    for (auto& recv_buffer : recv_buffers) {
      MemoryInspector::GetInstance().deallocate(recv_buffer.size() *
                                                sizeof(MESSAGE_T));
    }
    t1_gather_calc_ += GetCurrentTime();
  }

  template <typename GRAPH_T, typename MESSAGE_T>
  void ScatterMasterVertices(
      const GRAPH_T& frag,
      const typename GRAPH_T::template vertex_array_t<MESSAGE_T>& input,
      typename GRAPH_T::template both_vertex_array_t<MESSAGE_T>& output) {
    if (!std::is_pod<MESSAGE_T>::value) {
      LOG(FATAL) << "not implemented for non-POD type";
    }
    const auto& input_range = input.GetVertexRange();
    const MESSAGE_T* input_data = &input[*input_range.begin()];
    size_t input_size = input_range.size();

    const auto& partitioner = frag.GetPartitioner();
    std::vector<MPI_Request> requests;
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t dst_fid = (fid_ + i) % fnum_;
      auto dst_vertices = partitioner.get_dst_vertices(dst_fid);
      if (input_range.IsSubsetOf(dst_vertices)) {
        sync_comm::isend_buffer(input_data, input_size, dst_fid, 0, comm_,
                                requests);
        sent_size_ += input_size * sizeof(MESSAGE_T);
        continue;
      } else {
        CHECK(!input_range.OverlapWith(dst_vertices));
      }
      auto src_vertices = partitioner.get_src_vertices(dst_fid);
      if (input_range.IsSubsetOf(src_vertices)) {
        sync_comm::isend_buffer(input_data, input_size, dst_fid, 0, comm_,
                                requests);
        sent_size_ += input_size * sizeof(MESSAGE_T);
        continue;
      } else {
        CHECK(!input_range.OverlapWith(src_vertices));
      }
    }

    const auto& output_range = output.GetVertexRange();
    for (fid_t i = 1; i < fnum_; ++i) {
      fid_t src_fid = (fid_ + fnum_ - i) % fnum_;
      auto master_vertices = partitioner.get_master_vertices(src_fid);
      if (master_vertices.IsSubsetOf(output_range)) {
        MESSAGE_T* output_data = &output[*master_vertices.begin()];
        size_t output_size = master_vertices.size();
        sync_comm::recv_buffer(output_data, output_size, src_fid, 0, comm_);
      } else {
        CHECK(!master_vertices.OverlapWith(output_range));
      }
    }

    if (input_range.IsSubsetOf(partitioner.get_master_vertices(fid_))) {
      MESSAGE_T* output_data = &output[*input_range.begin()];
      std::copy(input_data, input_data + input_size, output_data);
    } else {
      CHECK(!input_range.OverlapWith(partitioner.get_master_vertices(fid_)));
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

 private:
  fid_t fid_;
  fid_t fnum_;
  CommSpec comm_spec_;

  MPI_Comm comm_;
  int round_;

  size_t sent_size_;

  bool force_terminate_;
  TerminateInfo terminate_info_;

  double t0_gather_comm_ = 0;
  double t1_gather_calc_ = 0;
};

}  // namespace grape

#endif  // GRAPE_PARALLEL_GATHER_SCATTER_MESSAGE_MANAGER_H_
