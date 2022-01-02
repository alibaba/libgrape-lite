#ifndef GRAPE_GPU_COMMUNICATION_DRIVER_H_
#define GRAPE_GPU_COMMUNICATION_DRIVER_H_

#include <mpi.h>

#include <cassert>
#include <type_traits>
#include <iostream>

#include "grape_gpu/config.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/segment.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/shared_value.h"

namespace grape_gpu {

enum class SyncMode { SM_ASYNC, SM_SYNC };
const SyncMode ASYNC = SyncMode::SM_ASYNC;
const SyncMode SYNC  = SyncMode::SM_SYNC;

typedef struct MetaInfo{
  size_t size;
  size_t dstGPU;
  MetaInfo() {}
  MetaInfo(size_t size, size_t dstGPU) : size(size), dstGPU(dstGPU) {}
} MetaInfo;

template<typename T>
class Pipeline{
  public:
    Pipeline(size_t chunk_size, size_t chunk_num) :
      buffers_(chunk_num), 
      mem_handles_(chunk_num),
      read_events_(chunk_num),
      write_events_(chunk_num), 
      read_event_handles_(chunk_num),
      write_event_handles_(chunk_num),
      chunk_size_(chunk_size), 
      head_(0) {}
    
    void ConsumeOne() { head_ = (head_ + 1) % buffers_.size(); } 

    DEV_HOST_INLINE T* GetAddr(size_t i) { return buffers_[i]; }
    cudaEvent_t GetReadEvent(size_t i) { return read_events_[i]; }
    cudaEvent_t GetWriteEvent(size_t i) { return write_events_[i]; }

    //Pipeline(const Pipeline&) = delete;
    //Pipeline(const Pipeline&&) {}
    //Pipeline& operator=(const Pipeline&) = delete;
    virtual ~Pipeline() = default;
  protected:
    std::vector<T*> buffers_;
    std::vector<cudaEvent_t> read_events_;
    std::vector<cudaEvent_t> write_events_;
    std::vector<cudaIpcMemHandle_t> mem_handles_;
    // event-based sync for CPU
    std::vector<cudaIpcEventHandle_t> read_event_handles_;
    std::vector<cudaIpcEventHandle_t> write_event_handles_;
    // ring-buffer-based sync for GPU
    size_t chunk_size_;
    size_t head_; // head maintained by host
};

template<typename T>
class PipelineReceiver : public Pipeline <T> {
  public:
    PipelineReceiver(size_t chunk_size, size_t chunk_num) :
      Pipeline<T>(chunk_size, chunk_num) {
      //VLOG(2) << "chunk info: " << chunk_size << " " << chunk_num;
      for (size_t i = 0; i < this->buffers_.size(); ++i) {
        CHECK_CUDA(cudaMalloc((void**)&this->buffers_[i], chunk_size * sizeof(T)));
        CHECK_CUDA(cudaIpcGetMemHandle((cudaIpcMemHandle_t*) &this->mem_handles_[i], 
                   this->buffers_[i]));
        CHECK_CUDA(cudaEventCreate(&this->read_events_[i], 
                   cudaEventDisableTiming | cudaEventInterprocess));
        CHECK_CUDA(cudaEventCreate(&this->write_events_[i], 
                   cudaEventDisableTiming | cudaEventInterprocess));
        CHECK_CUDA(cudaIpcGetEventHandle(
                  (cudaIpcEventHandle_t *) &this->read_event_handles_[i], 
                  this->read_events_[i]));
        CHECK_CUDA(cudaIpcGetEventHandle(
                  (cudaIpcEventHandle_t *) &this->write_event_handles_[i], 
                  this->write_events_[i]));
      }
    }

    // single node only
    void Init(int rank, int idx, const MPI_Comm& comm) {
      if(rank == idx) return;
      char* send_buf = reinterpret_cast<char*>(&this->mem_handles_[0]);
      int size = sizeof(cudaIpcMemHandle_t) * this->mem_handles_.size();
      MPI_Send(send_buf, size, MPI_CHAR, idx, rank, comm);

      send_buf = reinterpret_cast<char*>(&this->read_event_handles_[0]);
      size = sizeof(cudaIpcEventHandle_t) * this->read_event_handles_.size();
      MPI_Send(send_buf, size, MPI_CHAR, idx, rank, comm);

      send_buf = reinterpret_cast<char*>(&this->write_event_handles_[0]);
      size = sizeof(cudaIpcEventHandle_t) * this->write_event_handles_.size();
      MPI_Send(send_buf, size, MPI_CHAR, idx, rank, comm);
    }

    bool Probe() {
      auto status = cudaEventQuery(this->read_events_[this->head_]);
      if (cudaSuccess == status) return true;
      else if (cudaErrorNotReady == status) return false;
      else CHECK_CUDA(status);
      return false;
    }

    MetaInfo ReceiveMeta(const Stream& stream) {
      MetaInfo mi;
      auto head = this->head_;
      char* ptr = reinterpret_cast<char*>(this->buffers_[head] + this->chunk_size_);
      ptr = ptr - sizeof(MetaInfo);
      char* dst = reinterpret_cast<char*>(&mi);
      CHECK_CUDA(cudaMemcpyAsync(dst, ptr, sizeof(MetaInfo),
                                 cudaMemcpyDeviceToHost, stream.cuda_stream()));
      CHECK_CUDA(cudaStreamSynchronize(stream.cuda_stream()));
      return mi;
    }

    Segment<T> Receive (size_t msg_size, const Stream& stream) {
      auto head = this->head_;
      assert(Probe()); // potential overhead?
      CHECK_CUDA(cudaStreamWaitEvent(stream.cuda_stream(), this->read_events_[head], 0));
      int64_t recv_handle = head;
      void* meta = reinterpret_cast<void*>(recv_handle);
      Segment<T> seg(this->buffers_[head], msg_size, meta);
      this->ConsumeOne();
      return seg;
    }
    
    // Restore the segment
    void Restore(Segment<T>& seg, const Stream& stream) {
      int64_t rh = reinterpret_cast<int64_t>(seg.metadata);
      CHECK_CUDA(cudaEventRecord(this->write_events_[rh], stream.cuda_stream()));
    }

    void Release(int rank, int idx) {
      if(rank == idx) return ;
      for(size_t i = 0; i < this->buffers_.size(); ++i) {
        CHECK_CUDA(cudaFree(this->buffers_[i]));
        CHECK_CUDA(cudaEventDestroy(this->read_events_[i]));
        CHECK_CUDA(cudaEventDestroy(this->write_events_[i]));
      }
    }
};

template<typename T>
class PipelineSender : public Pipeline <T> {
  public:
    PipelineSender(size_t chunk_size, size_t chunk_num) :
      Pipeline<T>(chunk_size, chunk_num) {}

    void Init(int rank, int idx, const MPI_Comm& comm) {
      if(rank == idx) return;
      char* recv_buf = reinterpret_cast<char*>(&this->mem_handles_[0]);
      int size = sizeof(cudaIpcMemHandle_t) * this->mem_handles_.size();
      MPI_Recv(recv_buf, size, MPI_CHAR, idx, idx, comm, NULL);

      recv_buf = reinterpret_cast<char*>(&this->read_event_handles_[0]);
      size = sizeof(cudaIpcEventHandle_t) * this->read_event_handles_.size();
      MPI_Recv(recv_buf, size, MPI_CHAR, idx, idx, comm, NULL);

      recv_buf = reinterpret_cast<char*>(&this->write_event_handles_[0]);
      size = sizeof(cudaIpcEventHandle_t) * this->read_event_handles_.size();
      MPI_Recv(recv_buf, size, MPI_CHAR, idx, idx, comm, NULL);

      for(int i=0; i<this->mem_handles_.size(); ++i) {
        CHECK_CUDA(cudaIpcOpenMemHandle((void**)&this->buffers_[i], *(cudaIpcMemHandle_t *)&this->mem_handles_[i],
                   cudaIpcMemLazyEnablePeerAccess));
        CHECK_CUDA(cudaIpcOpenEventHandle(&this->read_events_[i], *(cudaIpcEventHandle_t *)&this->read_event_handles_[i]));
        CHECK_CUDA(cudaIpcOpenEventHandle(&this->write_events_[i], *(cudaIpcEventHandle_t *)&this->write_event_handles_[i]));
      }
    }

    // return a event handle
    int Send(T* msg, size_t size,  MetaInfo mi, const Stream &stream) {
      auto head = this->head_;
      //assert(size <= chunk_size);
      CHECK_CUDA(cudaStreamWaitEvent(stream.cuda_stream(), this->write_events_[head], 0));
      CHECK_CUDA(cudaMemcpyAsync((void*) msg, (void*) this->buffers_[head], 
                 size * sizeof(T), cudaMemcpyDeviceToDevice, stream.cuda_stream()));
      //FIXME redundant memcpy kernel (for metadata)
      char* ptr = reinterpret_cast<char*>(this->buffers_[head] + this->chunk_size_);
      ptr = ptr - sizeof(MetaInfo);
      CHECK_CUDA(cudaMemcpyAsync((void*) msg, (void*) ptr, 
                 size * sizeof(T), cudaMemcpyDeviceToDevice, stream.cuda_stream()));
      CHECK_CUDA(cudaEventRecord(this->read_events_[head], stream.cuda_stream()));
      int send_handle = head;
      this->ConsumeOne();
      return send_handle;
    }

    // query whether send is over
    bool QuerySend(int sh) {
      return cudaSuccess == cudaEventQuery(this->read_events_[sh]);
    }

    // wait until send is over
    void WaitSend(int sh, const Stream& stream) {
      CHECK_CUDA(cudaStreamWaitEvent(stream.cuda_stream(), this->read_events_[sh], 0));
    }

    void Release(int rank, int idx) {
      if (rank == idx) return;
      for (size_t i = 0; i < this->buffers_.size(); ++i) {
        CHECK_CUDA(cudaIpcCloseMemHandle(this->buffers_[i]));
        CHECK_CUDA(cudaEventDestroy(this->read_events_[i]));
        CHECK_CUDA(cudaEventDestroy(this->write_events_[i]));
      }
    }
};

template<SyncMode S, typename T>
class Driver{};

template<typename T>
class Driver<SyncMode::SM_ASYNC, T> {
  public:
    Driver() {}

    void Init(const grape::CommSpec& comm_spec, size_t chunk_size, size_t chunk_num){
      this->comm_spec_ = comm_spec;
      this->chunk_size_ = chunk_size;
      this->chunk_num_ = chunk_num;
      int rank = comm_spec_.local_id();
      int size = comm_spec_.local_num();
      MPI_Comm comm = comm_spec_.local_comm();
      
      for(int i=0; i<size; ++i){
        recv_pipelines_.emplace_back(chunk_size, chunk_num_);
        send_pipelines_.emplace_back(chunk_size, chunk_num_);
      }
      idx_ = 0;

      // swing exchange
      for (int i = 0; i < recv_pipelines_.size(); ++i) {
        if (rank < i) {
          recv_pipelines_[i].Init(rank, i, comm);
          send_pipelines_[i].Init(rank, i, comm);
        } else {
          send_pipelines_[i].Init(rank, i, comm);
          recv_pipelines_[i].Init(rank, i, comm);
        }
      }
    }

    size_t Size() { return recv_pipelines_.size(); }

    bool Query(size_t r) { 
      return recv_pipelines_[r].Probe(); 
    }
    
    Segment<T> ReceiveFrom(size_t r, const Stream& stream) {
      // we should passed the polling at this time.
      // Assert polling may incur the performance
      auto mi = recv_pipelines_[r].ReceiveMeta(stream);
      auto seg = recv_pipelines_[r].Receive(mi.size, stream);
      if(mi.dstGPU != r) { // transmit
        int sh = SendTo(mi.dstGPU, seg.GetSegmentPtr(), seg.GetTotalSize(), stream);
        this->WaitSend(sh, stream);
        this->Restore(seg, stream);
        // FIXME: now compute thread should handle a empty segment
        return Segment<T>(); 
      }
      int64_t rh = reinterpret_cast<int64_t>(seg.metadata) 
                 + r * chunk_num_;
      seg.metadata = reinterpret_cast<void*>(rh);
      return seg;
    }

    // return true if we have stuff.
    bool Polling(size_t& r) {
      size_t size = recv_pipelines_.size();
      for(int i = 0; i < size; i ++) {
        int pos = (idx_ + i) % size;
        if(recv_pipelines_[pos].Probe()) {
          return true;
        }
      }
      return false;
    }

    Segment<T> PollingUntilReceive(size_t& r, const Stream& stream) {
      size_t size = recv_pipelines_.size();
      for (int i = 0;; i++) {// busy-waiting
        assert(i<100000);
        int pos = (idx_ + i) % size;
        if (recv_pipelines_[pos].Probe()) {
            idx_ = pos;
            auto seg = ReceiveFrom(pos, stream);
            if (0 == seg.GetTotalSize()) continue;
            r = pos;
            return seg;
        }
      }
      assert(true); //should never touch this line
    }

    // return the buffer
    void Restore(Segment<T>& seg, const Stream& stream) {
      int rh = reinterpret_cast<int64_t>(seg.metadata);
      int r = rh / chunk_num_;
      int h = rh % chunk_num_;
      seg.metadata = reinterpret_cast<void*>(h);
      recv_pipelines_[r].Restore(seg, stream);
    }

    // compute thread will use send_handle to check whether the copy is over
    template<typename MSG_T>
    int SendTo(size_t r, MSG_T* msg, size_t size, const Stream &stream) {
      char* _msg = reinterpret_cast<char*>(msg);
      size_t _size = size*sizeof(MSG_T);
      int me = comm_spec_.local_id();
      MetaInfo mi(size, r);
      if(!CheckRoute(r, comm_spec_.local_id())) {
        r = -route_[me][r]; // transit
        mi.dstGPU = r;
      }
      int sh = send_pipelines_[r].Send(_msg, _size, mi, stream);
      return r * chunk_num_ + sh;
    }

    bool QuerySend(int sh, const Stream& stream){
      size_t r = sh / chunk_num_;
      size_t h = sh % chunk_num_;
      return send_pipelines_[r].QuerySend(h, stream);
    }

    void WaitSend(int sh, const Stream& stream){
      size_t r = sh / chunk_num_;
      size_t h = sh % chunk_num_;
      send_pipelines_[r].WaitSend(h, stream);
    }

    void Release () {
      int rank = comm_spec_.local_id();
      for(int i = 0; i < send_pipelines_.size(); ++i) {
        send_pipelines_[i].Release(rank, i);
      }
      for(int i = 0; i < recv_pipelines_.size(); ++i) {
        recv_pipelines_[i].Release(rank, i);
      }
    }

    bool CheckRoute(int i, int j) {
      return route_[i][j] > 0;
    }

    void BuildRoute() {
      int size = comm_spec_.local_num();
      MPI_Comm comm = comm_spec_.local_comm();
      for(int i = 0; i < size; ++i) {
        route_[i].clear();
        for(int j = 0; j < size; ++j) {
          int accessSupported = 0, perfRank = 0;
          if(i != j){
            CHECK_CUDA(cudaDeviceGetP2PAttribute(&accessSupported, 
                           cudaDevP2PAttrAccessSupported, i, j));
            CHECK_CUDA(cudaDeviceGetP2PAttribute(&perfRank, 
                           cudaDevP2PAttrPerformanceRank, i, j));
          } 
          int quality = accessSupported * (accessSupported + perfRank == 0);
          route_[i].push_back(quality);
        }
      }

      for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
          if(route_[i][j] == 0) {
            for(int k=0; k < size; ++k) {
              if(route_[i][k] > 0 && route_[k][j] > 0){
                route_[i][j] = -k;
                break;
              }
            }
          }
        }
      }
    }

    virtual ~Driver() = default;
  private:
    std::vector<PipelineSender<T>> send_pipelines_; // 8
    std::vector<PipelineReceiver<T>> recv_pipelines_; // 8
    grape::CommSpec comm_spec_;
    std::vector<std::vector<int>> route_;
    size_t idx_; // for fairness
    size_t chunk_size_;
    size_t chunk_num_;
};

} // namespace graph_gpu


#endif //GRAPE_GPU_COMMUNICATION_DRIVER_H_

