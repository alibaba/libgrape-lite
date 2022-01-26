/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef GRAPE_CUDA_COMMUNICATION_COMMUNICATOR_H_
#define GRAPE_CUDA_COMMUNICATION_COMMUNICATOR_H_
#include <mpi.h>
#include <nccl.h>

#include <type_traits>

#include "grape/config.h"
#include "grape/cuda/utils/shared_value.h"
#include "grape/cuda/utils/stream.h"

namespace grape {
namespace cuda {
class Communicator {
 public:
  Communicator() = default;

  virtual ~Communicator() = default;

  void InitCommunicator(MPI_Comm comm, ncclComm_t nccl_comm) {
    comm_ = comm;
    nccl_comm_ = nccl_comm;
    CHECK_NCCL(ncclCommCount(nccl_comm, &n_rank_));
  }

  template <typename T>
  void Sum(T msg_in, T& msg_out, const Stream& stream) {
    reduce(msg_in, msg_out, ncclSum, stream);
  }

  template <typename T>
  void Sum(T* msg_in, T& msg_out, const Stream& stream) {
    reduce(msg_in, msg_out, ncclSum, stream);
  }

  template <typename T>
  void Sum(T msg_in, T& msg_out) {
    reduce(&msg_in, &msg_out, MPI_SUM);
  }

  template <typename T>
  void Min(T msg_in, T& msg_out, const Stream& stream) {
    reduce(msg_in, msg_out, ncclMin, stream);
  }

  template <typename T>
  void Min(T* msg_in, T& msg_out, const Stream& stream) {
    reduce(msg_in, msg_out, ncclMin, stream);
  }

  template <typename T>
  void Min(T msg_in, T& msg_out) {
    reduce(&msg_in, &msg_out, MPI_MIN);
  }

  template <typename T>
  void Max(T msg_in, T& msg_out, const Stream& stream) {
    reduce(msg_in, msg_out, ncclMax, stream);
  }

  template <typename T>
  void Max(T* msg_in, T& msg_out, const Stream& stream) {
    reduce(msg_in, msg_out, ncclMax, stream);
  }

  template <typename T>
  void Max(T msg_in, T& msg_out) {
    reduce(&msg_in, &msg_out, MPI_MAX);
  }

  template <typename T>
  std::vector<T> AllGather(T msg_in) {
    int worker_num;

    MPI_Comm_size(comm_, &worker_num);
    std::vector<T> out(worker_num);

    MPI_Allgather(&msg_in, 1, mpiType<T>(), out.data(), 1, mpiType<T>(), comm_);
    return out;
  }

 private:
  template <typename T>
  inline ncclDataType_t ncclType() {
    static_assert(!std::is_same<T, bool>::value && std::is_arithmetic<T>::value,
                  "Unsupported type");

    ncclDataType_t type;

    if (std::is_same<T, char>::value) {
      type = ncclChar;
    } else if (std::is_same<T, int8_t>::value) {
      type = ncclInt8;
    } else if (std::is_same<T, uint8_t>::value) {
      type = ncclUint8;
    } else if (std::is_same<T, int>::value) {
      type = ncclInt;
    } else if (std::is_same<T, int32_t>::value) {
      type = ncclInt32;
    } else if (std::is_same<T, uint32_t>::value) {
      type = ncclUint32;
    } else if (std::is_same<T, int64_t>::value) {
      type = ncclInt64;
    } else if (std::is_same<T, uint64_t>::value) {
      type = ncclUint64;
    } else if (std::is_same<T, float>::value) {
      type = ncclFloat;
    } else if (std::is_same<T, double>::value) {
      type = ncclDouble;
    }
    return type;
  }

  template <typename T>
  inline MPI_Datatype mpiType() {
    static_assert(!std::is_same<T, bool>::value && std::is_arithmetic<T>::value,
                  "Unsupported type");

    MPI_Datatype type;

    if (std::is_same<T, char>::value) {
      type = MPI_CHAR;
    } else if (std::is_same<T, int8_t>::value) {
      type = MPI_INT8_T;
    } else if (std::is_same<T, uint8_t>::value) {
      type = MPI_UINT8_T;
    } else if (std::is_same<T, int>::value) {
      type = MPI_INT;
    } else if (std::is_same<T, int32_t>::value) {
      type = MPI_INT32_T;
    } else if (std::is_same<T, uint32_t>::value) {
      type = MPI_UINT32_T;
    } else if (std::is_same<T, int64_t>::value) {
      type = MPI_INT64_T;
    } else if (std::is_same<T, uint64_t>::value) {
      type = MPI_UINT64_T;
    } else if (std::is_same<T, float>::value) {
      type = MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
      type = MPI_DOUBLE;
    }
    return type;
  }

  template <typename T>
  void reduce(T* msg_in, T* msg_out, ncclRedOp_t op, const Stream& stream) {
    if (n_rank_ == 1) {
      CHECK_CUDA(cudaMemcpyAsync(msg_out, msg_in, sizeof(T), cudaMemcpyDefault,
                                 stream.cuda_stream()));
    } else {
      auto type = ncclType<T>();

      CHECK_NCCL(ncclAllReduce(msg_in, msg_out, sizeof(T), type, op, nccl_comm_,
                               stream.cuda_stream()));
    }
    stream.Sync();
  }

  template <typename T>
  void reduce(T msg_in, T& msg_out, ncclRedOp_t op, const Stream& stream) {
    pinned_vector<T> in(1);
    pinned_vector<T> out(1);

    in[0] = msg_in;
    reduce(thrust::raw_pointer_cast(in.data()),
           thrust::raw_pointer_cast(out.data()), op, stream);
    msg_out = out[0];
  }

  template <typename T>
  void reduce(T* msg_in, T& msg_out, ncclRedOp_t op, const Stream& stream) {
    pinned_vector<T> out(1);

    reduce(msg_in, thrust::raw_pointer_cast(out.data()), op, stream);
    msg_out = out[0];
  }

  template <typename T>
  void reduce(const T* msg_in, T* msg_out, MPI_Op op) {
    MPI_Allreduce(msg_in, msg_out, 1, mpiType<T>(), op, comm_);
  }

 private:
  int n_rank_{};
  ncclComm_t nccl_comm_{};
  MPI_Comm comm_;
};

template <typename APP_T>
typename std::enable_if<std::is_base_of<Communicator, APP_T>::value>::type
InitCommunicator(std::shared_ptr<APP_T> app, MPI_Comm comm,
                 ncclComm_t nccl_comm) {
  app->InitCommunicator(comm, nccl_comm);
}

template <typename APP_T>
typename std::enable_if<!std::is_base_of<Communicator, APP_T>::value>::type
InitCommunicator(std::shared_ptr<APP_T>, MPI_Comm, ncclComm_t) {}
}  // namespace cuda
}  // namespace grape
#endif  // GRAPE_CUDA_COMMUNICATION_COMMUNICATOR_H_
