#ifndef GRAPEGPU_EXAMPLES_PC_QUEUE_H_
#define GRAPEGPU_EXAMPLES_PC_QUEUE_H_
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "grape_gpu/utils/event.h"
#include "grape_gpu/utils/pc_queue.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/stream.h"
#include "gtest/gtest.h"
namespace grape_gpu {

template <bool Prepend = false, bool Warp = true>
__global__ void ProduceKernel(dev::PCQueue<int> queue, const int* work,
                              int work_size) {
  int tid = TID_1D;

  if (tid < work_size) {
    if (Prepend) {
      if (Warp) {
        queue.PrependWarp(work[tid]);
      } else {
        queue.Prepend(work[tid]);
      }
    } else {
      if (Warp) {
        queue.AppendWarp(work[tid]);
      } else {
        queue.Append(work[tid]);
      }
    }
  }
}

void TestAppendPrependPop(int nappend, int nprepend, int wl_alloc_factor = 1,
                          int chunk_size_factor = 100) {
  srand(static_cast<unsigned>(22522));

  int queue_capacity = (nappend + nprepend) / wl_alloc_factor;
  int max_chunk_size = std::max((nappend + nprepend) / chunk_size_factor, 1);

  CHECK_CUDA(cudaSetDevice(0));

  thrust::host_vector<int> append_input(nappend);
  thrust::host_vector<int> prepend_input(nprepend);
  SharedValue<int> sum;

  int regression_sum = 0;

  for (int i = 0; i < nappend; ++i) {
    regression_sum += (append_input[i] = (rand() % 100));
  }
  for (int i = 0; i < nprepend; ++i) {
    regression_sum += (prepend_input[i] = (rand() % 100));
  }

  thrust::device_vector<int> d_append_input = append_input;
  thrust::device_vector<int> d_prepend_input = prepend_input;

  PCQueue<int> pcqueue(queue_capacity);

  // sync objects
  std::mutex mutex;
  std::condition_variable cv;
  bool exit = false;
  Event signal;

  Stream producer_stream;
  pcqueue.ResetAsync(producer_stream);  // init

  producer_stream.Sync();  // sync

  std::thread worker([&]() {
    Stream alternating_stream;

    srand(static_cast<unsigned>(22422));
    int pos = 0;

    while (true) {
      // prepend some work
      if (pos < nprepend) {
        int chunk = std::min(nprepend - pos, rand() % (max_chunk_size - 1) + 1);

        dim3 block_dims(256, 1, 1);
        dim3 grid_dims(round_up(chunk, block_dims.x), 1, 1);

        ProduceKernel<true>
            <<<grid_dims, block_dims, 0, alternating_stream.cuda_stream()>>>(
                pcqueue._DeviceObject(),
                thrust::raw_pointer_cast(d_prepend_input.data()) + pos, chunk);

        pos += chunk;
      }

      // check if worklist has work
      auto segs = pcqueue.GetSegs(alternating_stream);

      if (segs.empty()) {
        // wait for append producer
        {
          std::unique_lock<std::mutex> guard(mutex);
          signal.Wait(alternating_stream);
          segs = pcqueue.GetSegs(alternating_stream);

          while (segs.empty()) {
            if (exit)
              break;
            cv.wait(guard);
            signal.Wait(alternating_stream);
            segs = pcqueue.GetSegs(alternating_stream);
          }
        }
      }

      if (segs.empty())
        break;

      // do and pop the work
      for (auto& seg : segs) {
        auto work_size = seg.GetSegmentSize();
        auto* work = seg.GetSegmentPtr();
        auto* d_sum = sum.data();

        LaunchKernel(alternating_stream, work_size,
                     [work, work_size, d_sum] __device__() {
                       for (int tid = TID_1D; tid < work_size;
                            tid += TOTAL_THREADS_1D) {
                         atomicAdd(d_sum, work[tid]);
                       }
                     });

        pcqueue.PopAsync(alternating_stream, seg.GetSegmentSize());
      }
    }

    alternating_stream.Sync();
  });

  int pos = 0;

  while (pos < nappend) {
    int chunk = std::min(nappend - pos, rand() % (max_chunk_size));

    dim3 producer_block_dims(256, 1, 1);
    dim3 producer_grid_dims(round_up(chunk, producer_block_dims.x), 1, 1);

    ProduceKernel<<<producer_grid_dims, producer_block_dims, 0,
                    producer_stream.cuda_stream()>>>(
        pcqueue._DeviceObject(),
        thrust::raw_pointer_cast(d_append_input.data()) + pos, chunk);

    pcqueue.CommitPendingAsync(producer_stream);

    auto ev = Event::Create();

    ev.Record(producer_stream);

    {
      std::lock_guard<std::mutex> lock(mutex);
      signal = ev;
      cv.notify_one();
    }

    pos += chunk;
  }

  producer_stream.Sync();

  {
    std::lock_guard<std::mutex> lock(mutex);
    exit = true;
    cv.notify_one();
  }

  worker.join();

  int output_sum = sum.get();

  ASSERT_EQ(regression_sum, output_sum);
}

void TestOverflow() {
  Stream stream;
  PCQueue<int> pc_queue(1000, "overflowed q1");

  pc_queue.ResetAsync(stream);

  int size = 2000;

  LaunchKernel(
      stream,
      [size] __device__(dev::PCQueue<int> dev_pcq) {
        for (int tid = TID_1D; tid < size; tid += TOTAL_THREADS_1D) {
          dev_pcq.AppendWarp(tid);
        }
      },
      pc_queue._DeviceObject());

  pc_queue.CommitPendingAsync(stream);

  LOG(INFO) << pc_queue.GetSpace(stream);
}

TEST(PCQueue, ProducerConsumer) {
  TestAppendPrependPop(2048, 0);
  TestAppendPrependPop(2048, 32);
  TestAppendPrependPop(0, 2048);
  TestAppendPrependPop(32, 2048);
  TestAppendPrependPop(2048, 2048);
  TestAppendPrependPop(2048 * 1024, 2048 * 2048);
  TestAppendPrependPop(2048 * 2048, 2048 * 1024);
  TestAppendPrependPop(2048 * 2048, 2048 * 2048);

//  TestOverflow();
}
}  // namespace grape_gpu
#endif  // GRAPEGPU_EXAMPLES_PC_QUEUE_H_
