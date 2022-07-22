#ifndef GRAPEGPU_EXAMPLES_IPC_H_
#define GRAPEGPU_EXAMPLES_IPC_H_
#include <iostream>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "grape_gpu/communication/cudaIPC.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/event.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/stream.h"
#include "gtest/gtest.h"

#define DEVICES (8)
#define DATA_SIZE (64ULL << 10ULL)

namespace grape_gpu {

__global__ void simpleKernel(char *ptr, int sz, char val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
    ptr[idx] = val;
  }
}

void TestWrite() {
  MPI_Init(NULL, NULL);
  int rank;
  int nums;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nums);
  CHECK_CUDA(cudaSetDevice(rank));
  Driver<SyncMode::CM_SYNC, char> driver(DATA_SIZE, nums);
  Stream stream;
  driver.Init();
  char* mybuf; 
  char* hostbuf;
  CHECK_CUDA(cudaMalloc(&mybuf, DATA_SIZE));
  CHECK_CUDA(cudaMallocHost(&hostbuf, DATA_SIZE));
  simpleKernel<<<256, 256, 0, stream.cuda_stream()>>>(mybuf, DATA_SIZE, rank);
  CHECK_CUDA(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  
  CHECK_CUDA(cudaMemcpyAsync(hostbuf, mybuf, DATA_SIZE, cudaMemcpyDeviceToHost, stream.cuda_stream()));
  CHECK_CUDA(cudaStreamSynchronize(stream.cuda_stream()));
  for (unsigned long long j=0; j < DATA_SIZE; ++j) {
    ASSERT_EQ(hostbuf[j], rank);
  }

  for(int i=1; i<nums; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    int peer  = (rank + i) % nums; 
    int pre   = (rank + nums - i) % nums;
    auto sptr = driver.GetAddrSend(peer);
    auto rptr = driver.GetAddrRecv(pre);
    auto sev  = driver.GetEventSend(peer);
    auto rev  = driver.GetEventRecv(pre);

    CHECK_CUDA(cudaMemcpyAsync(sptr, mybuf, DATA_SIZE, cudaMemcpyDeviceToDevice, stream.cuda_stream()));
    CHECK_CUDA(cudaEventRecord(sev, stream.cuda_stream()));

    CHECK_CUDA(cudaStreamWaitEvent(stream.cuda_stream(), rev, 0));
    CHECK_CUDA(cudaMemcpyAsync(hostbuf, rptr, DATA_SIZE, cudaMemcpyDeviceToHost, stream.cuda_stream()));

    CHECK_CUDA(cudaStreamSynchronize(stream.cuda_stream()));
    for (unsigned long long j=0; j < DATA_SIZE; ++j) {
      ASSERT_EQ(hostbuf[j], pre);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  driver.Release(rank);
  MPI_Finalize();
}

TEST(IPC, EXCHANGEPTR) {
  TestWrite();
}

} // grepe_gpu


#endif // GRAPEGPU_EXAMPLES_IPC_H_
