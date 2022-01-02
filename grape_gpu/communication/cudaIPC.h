#ifndef GRAPE_GPU_COMMUNICATION_DRIVER_H_
#define GRAPE_GPU_COMMUNICATION_DRIVER_H_

#include <mpi.h>

#include <cassert>
#include <type_traits>

#include "grape_gpu/config.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/stream.h"

namespace grape_gpu {

enum class SyncMode { CM_ASYNC, CM_SYNC };
const SyncMode ASYNC = SyncMode::CM_ASYNC;
const SyncMode SYNC  = SyncMode::CM_SYNC;

template<typename T>
class Pipeline{
  public:
    Pipeline() = default;
    
    DEV_HOST_INLINE T* GetAddr() { return buffer; }
    cudaEvent_t GetEvent() { return event; }

    virtual ~Pipeline() = default;
  protected:
    T* buffer;
    cudaEvent_t event;
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t event_handle;
};

template<typename T>
class PipelineReceiver : public Pipeline <T> {
  public:
    explicit PipelineReceiver(size_t size) {
      CHECK_CUDA(cudaMalloc((void**)&this->buffer, size * sizeof(T)));
      CHECK_CUDA(cudaIpcGetMemHandle((cudaIpcMemHandle_t*) &this->mem_handle,
                 this->buffer));
      CHECK_CUDA(cudaEventCreate(&this->event,
                 cudaEventDisableTiming | cudaEventInterprocess));
      CHECK_CUDA(cudaIpcGetEventHandle(
                (cudaIpcEventHandle_t *) &this->event_handle, this->event));
    }

    void Init(int rank, int idx) {
      if(rank == idx) return;
      char* send_buf = reinterpret_cast<char*>(&this->mem_handle);
      int size = sizeof(cudaIpcMemHandle_t);
      MPI_Send(send_buf, size, MPI_CHAR, idx, rank, MPI_COMM_WORLD);

      send_buf = reinterpret_cast<char*>(&this->event_handle);
      size = sizeof(cudaIpcEventHandle_t);
      MPI_Send(send_buf, size, MPI_CHAR, idx, rank, MPI_COMM_WORLD);
    }

    void Release(int rank, int idx) {
      if (rank == idx) return;
      CHECK_CUDA(cudaEventDestroy(this->event));
      CHECK_CUDA(cudaFree((void*)this->buffer));
    }
};

template<typename T>
class PipelineSender : public Pipeline <T> {
  public:
    explicit PipelineSender(size_t size) {}

    void Init(int rank, int idx) {
      if (rank == idx) return;
      char* recv_buf = reinterpret_cast<char*>(&this->mem_handle);
      int size = sizeof(cudaIpcMemHandle_t);
      MPI_Recv(recv_buf, size, MPI_CHAR, idx, idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      recv_buf = reinterpret_cast<char*>(&this->event_handle);
      size = sizeof(cudaIpcEventHandle_t);
      MPI_Recv(recv_buf, size, MPI_CHAR, idx, idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      CHECK_CUDA(cudaIpcOpenMemHandle((void**)&this->buffer, *(cudaIpcMemHandle_t *)&this->mem_handle,
                 cudaIpcMemLazyEnablePeerAccess));
      CHECK_CUDA(cudaIpcOpenEventHandle(&this->event, *(cudaIpcEventHandle_t *)&this->event_handle));
    }

    void Release(int rank, int idx) {
      if (rank == idx) return;
      CHECK_CUDA(cudaIpcCloseMemHandle(this->buffer));
      CHECK_CUDA(cudaEventDestroy(this->event));
    }
};

template<SyncMode S, typename T>
class Driver{};

template<typename T>
class Driver<SyncMode::CM_SYNC, T> {
  public:
    Driver(size_t size, size_t num_devices) {
      for(int i=0; i<num_devices; ++i){
        recv_pipelines.emplace_back(size);
        send_pipelines.emplace_back(size);
      }
    }

    void Init(){
      int rank, nums;
      MPI_Comm_size(MPI_COMM_WORLD, &nums);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // swing exchange
      for (int i = 0; i < recv_pipelines.size(); ++i) {
        if (rank < i) {
          recv_pipelines[i].Init(rank, i);
          send_pipelines[i].Init(rank, i);
        } else {
          send_pipelines[i].Init(rank, i);
          recv_pipelines[i].Init(rank, i);
        }
      }
    }

    T* GetAddrSend(int r) {
      return send_pipelines[r].GetAddr();
    }

    T* GetAddrRecv(int r) {
      return recv_pipelines[r].GetAddr();
    }

    cudaEvent_t GetEventSend(int r) {
      return send_pipelines[r].GetEvent();
    }

    cudaEvent_t GetEventRecv(int r) {
      return recv_pipelines[r].GetEvent();
    }

    void Release(int rank) {
      for(int i=0; i<send_pipelines.size(); ++i) send_pipelines[i].Release(rank, i);
      for(int i=0; i<send_pipelines.size(); ++i) recv_pipelines[i].Release(rank, i);
    }

    virtual ~Driver() = default;
  private:
    std::vector<PipelineSender<T>> send_pipelines; // 8
    std::vector<PipelineReceiver<T>> recv_pipelines; // 8
};

} // namespace graph_gpu


#endif //GRAPE_GPU_COMMUNICATION_DRIVER_H_
