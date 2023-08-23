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

#ifndef GRAPE_WORKER_COMM_SPEC_H_
#define GRAPE_WORKER_COMM_SPEC_H_

#include <mpi.h>
#include <stdlib.h>

#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/config.h"

namespace grape {

/**
 * @brief CommSpec records the mappings of fragments, workers, and the
 * threads(tasks) in each worker.
 *
 */
class CommSpec {
 public:
  CommSpec()
      : worker_num_(1),
        worker_id_(0),
        local_num_(1),
        local_id_(0),
        fid_(0),
        fnum_(1),
        comm_(NULL_COMM),
        owner_(false) {}

  CommSpec(const CommSpec& comm_spec)
      : worker_num_(comm_spec.worker_num_),
        worker_id_(comm_spec.worker_id_),
        local_num_(comm_spec.local_num_),
        local_id_(comm_spec.local_id_),
        fid_(comm_spec.fid_),
        fnum_(comm_spec.fnum_),
        comm_(comm_spec.comm_),
        owner_(false) {}

  ~CommSpec() {
    if (owner_ && ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
  }

  CommSpec& operator=(const CommSpec& rhs) {
    if (owner_ && ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }

    worker_num_ = rhs.worker_num_;
    worker_id_ = rhs.worker_id_;
    local_num_ = rhs.local_num_;
    local_id_ = rhs.local_id_;
    fid_ = rhs.fid_;
    fnum_ = rhs.fnum_;
    comm_ = rhs.comm_;
    owner_ = false;

    return *this;
  }

  void Init(MPI_Comm comm) {
    MPI_Comm_rank(comm, &worker_id_);
    MPI_Comm_size(comm, &worker_num_);

    comm_ = comm;
    owner_ = false;

    initLocalInfo();

    fnum_ = worker_num_;
    fid_ = worker_id_;
  }

  void Dup() {
    MPI_Comm old_comm = comm_;
    MPI_Comm_dup(old_comm, &comm_);
    owner_ = true;
  }

  inline int FragToWorker(fid_t fid) const { return static_cast<int>(fid); }

  inline fid_t WorkerToFrag(int wid) const { return static_cast<fid_t>(wid); }

  inline int worker_num() const { return worker_num_; }

  inline int worker_id() const { return worker_id_; }

  inline int local_num() const { return local_num_; }

  inline int local_id() const { return local_id_; }

  inline fid_t fnum() const { return fnum_; }

  inline fid_t fid() const { return fid_; }

  inline MPI_Comm comm() const { return comm_; }

 private:
  void initLocalInfo() {
    char hn[MPI_MAX_PROCESSOR_NAME];
    int hn_len;

    MPI_Get_processor_name(hn, &hn_len);

    char* recv_buf = reinterpret_cast<char*>(calloc(worker_num_, sizeof(hn)));
    MPI_Allgather(hn, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, recv_buf,
                  MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm_);

    std::vector<std::string> worker_host_names(worker_num_);
    for (int i = 0; i < worker_num_; ++i) {
      worker_host_names[i].assign(
          &recv_buf[i * MPI_MAX_PROCESSOR_NAME],
          strlen(&recv_buf[i * MPI_MAX_PROCESSOR_NAME]));
    }
    free(recv_buf);

    local_num_ = 0;
    local_id_ = -1;
    for (int i = 0; i < worker_num_; ++i) {
      if (i == worker_id_) {
        local_id_ = local_num_;
      }
      if (worker_host_names[i] == worker_host_names[worker_id_]) {
        ++local_num_;
      }
    }
  }

  int worker_num_;
  int worker_id_;

  int local_num_;
  int local_id_;

  fid_t fid_;
  fid_t fnum_;

  MPI_Comm comm_;
  bool owner_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_COMM_SPEC_H_
