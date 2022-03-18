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
  __attribute__((no_sanitize_address)) CommSpec()
      : worker_num_(1),
        worker_id_(0),
        local_num_(1),
        local_id_(0),
        fid_(0),
        fnum_(1),
        comm_(NULL_COMM),
        local_comm_(NULL_COMM),
        owner_(false),
        local_owner_(false) {}

  __attribute__((no_sanitize_address)) CommSpec(const CommSpec& comm_spec)
      : worker_num_(comm_spec.worker_num_),
        worker_id_(comm_spec.worker_id_),
        local_num_(comm_spec.local_num_),
        local_id_(comm_spec.local_id_),
        fid_(comm_spec.fid_),
        fnum_(comm_spec.fnum_),
        comm_(comm_spec.comm_),
        local_comm_(comm_spec.local_comm_),
        owner_(false),
        local_owner_(false),
        worker_host_id_(comm_spec.worker_host_id_),
        host_worker_list_(comm_spec.host_worker_list_) {}

  __attribute__((no_sanitize_address)) ~CommSpec() {
    if (owner_ && ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
    if (local_owner_ && ValidComm(local_comm_)) {
      MPI_Comm_free(&local_comm_);
    }
  }

  __attribute__((no_sanitize_address)) CommSpec& operator=(
      const CommSpec& rhs) {
    if (owner_ && ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
    if (local_owner_ && ValidComm(local_comm_)) {
      MPI_Comm_free(&local_comm_);
    }

    worker_num_ = rhs.worker_num_;
    worker_id_ = rhs.worker_id_;
    local_num_ = rhs.local_num_;
    local_id_ = rhs.local_id_;
    fid_ = rhs.fid_;
    fnum_ = rhs.fnum_;
    comm_ = rhs.comm_;
    local_comm_ = rhs.local_comm_;
    owner_ = false;
    local_owner_ = false;

    return *this;
  }

  __attribute__((no_sanitize_address)) void Init(
      MPI_Comm comm, const std::string& hostname = "") {
    if (owner_ && ValidComm(comm_)) {
      MPI_Comm_free(&comm_);
    }
    if (local_owner_ && ValidComm(local_comm_)) {
      MPI_Comm_free(&local_comm_);
    }
    MPI_Comm_rank(comm, &worker_id_);
    MPI_Comm_size(comm, &worker_num_);

    comm_ = comm;
    owner_ = false;
    local_owner_ = false;

    initLocalInfo(hostname);

    fnum_ = worker_num_;
    fid_ = worker_id_;
  }

  __attribute__((no_sanitize_address)) void Dup() {
    if (!owner_) {
      MPI_Comm old_comm = comm_;
      MPI_Comm_dup(old_comm, &comm_);
      owner_ = true;
    }
    if (!local_owner_) {
      MPI_Comm old_local_comm = local_comm_;
      MPI_Comm_dup(old_local_comm, &local_comm_);
      local_owner_ = true;
    }
  }

  inline int FragToWorker(fid_t fid) const { return static_cast<int>(fid); }

  inline fid_t WorkerToFrag(int wid) const { return static_cast<fid_t>(wid); }

  inline int worker_num() const { return worker_num_; }

  inline int worker_id() const { return worker_id_; }

  inline int local_num() const { return local_num_; }

  inline int local_id() const { return local_id_; }

  inline fid_t fnum() const { return fnum_; }

  inline fid_t fid() const { return fid_; }

  __attribute__((no_sanitize_address)) inline MPI_Comm comm() const {
    return comm_;
  }

  __attribute__((no_sanitize_address)) inline MPI_Comm local_comm() const {
    return local_comm_;
  }
  inline int host_num() const { return host_worker_list_.size(); }

  inline int host_id() const { return worker_host_id_[worker_id_]; }

  inline int host_id(int worker_id) const { return worker_host_id_[worker_id]; }

  inline std::vector<int>& host_worker_list(int host_id) {
    return host_worker_list_[host_id];
  }

  inline const std::vector<int>& host_worker_list(int host_id) const {
    return host_worker_list_[host_id];
  }

 private:
  __attribute__((no_sanitize_address)) void initLocalInfo(
      const std::string& hostname) {
    char hn[MPI_MAX_PROCESSOR_NAME];
    int hn_len;

    if (hostname.empty()) {
      MPI_Get_processor_name(hn, &hn_len);
    } else {
      hn_len = hostname.length() > MPI_MAX_PROCESSOR_NAME
                   ? MPI_MAX_PROCESSOR_NAME
                   : hostname.length();
      strncpy(hn, hostname.c_str(), hn_len);
    }

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

    std::map<std::string, int> hostname2id;
    worker_host_id_.clear();
    worker_host_id_.resize(worker_num_);

    host_worker_list_.clear();

    for (int i = 0; i < worker_num_; ++i) {
      auto iter = hostname2id.find(worker_host_names[i]);
      if (iter == hostname2id.end()) {
        int new_id = hostname2id.size();
        worker_host_id_[i] = new_id;
        hostname2id[worker_host_names[i]] = new_id;

        std::vector<int> vec;
        vec.push_back(i);
        host_worker_list_.emplace_back(std::move(vec));
      } else {
        worker_host_id_[i] = iter->second;
        host_worker_list_[iter->second].push_back(i);
      }
    }
    if (local_owner_ && ValidComm(local_comm_)) {
      MPI_Comm_free(&local_comm_);
    }
    MPI_Comm_split(comm_, worker_host_id_[worker_id_], worker_id_,
                   &local_comm_);
    MPI_Comm_size(local_comm_, &local_num_);
    MPI_Comm_rank(local_comm_, &local_id_);
    local_owner_ = true;
  }

  int worker_num_;
  int worker_id_;

  int local_num_;
  int local_id_;

  fid_t fid_;
  fid_t fnum_;

  MPI_Comm comm_;
  MPI_Comm local_comm_;
  bool owner_;
  bool local_owner_;

  std::vector<int> worker_host_id_;
  std::vector<std::vector<int>> host_worker_list_;
};

}  // namespace grape

#endif  // GRAPE_WORKER_COMM_SPEC_H_
