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

#ifndef GRAPE_PARALLEL_PARALLEL_ENGINE_H_
#define GRAPE_PARALLEL_PARALLEL_ENGINE_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "grape/communication/sync_comm.h"
#include "grape/utils/thread_pool.h"
#include "grape/utils/vertex_set.h"
#include "grape/worker/comm_spec.h"
#include "grape/parallel/parallel_engine_spec.h"

namespace grape {
class ParallelEngine {
 public:
  ParallelEngine() : thread_num_(1) {}
  virtual ~ParallelEngine() {}

  void InitParallelEngine(
      const ParallelEngineSpec& spec = DefaultParallelEngineSpec()) {
    thread_num_ = spec.thread_num;
    thread_pool_.InitThreadPool(spec);
  }

  inline ThreadPool& GetThreadPool() { return thread_pool_; }

  /**
   * @brief Iterate a range specified by pointer pair concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param range The vertex range to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename T>
  inline void ForEach(const T* begin, const T* end,
                      const ITER_FUNC_T& iter_func) {
    size_t chunk_size = (end - begin) / thread_num_ + 1;

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] =
          thread_pool_.enqueue([chunk_size, &iter_func, begin, end, tid] {
            const T* cur_beg = std::min(begin + tid * chunk_size, end);
            const T* cur_end = std::min(begin + (tid + 1) * chunk_size, end);
            if (cur_beg != cur_end) {
              for (auto iter = cur_beg; iter != cur_end; ++iter) {
                iter_func(tid, iter);
              }
            }
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on vertexs of a VertexRange concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param range The vertex range to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const VertexRange<VID_T>& range,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    std::atomic<VID_T> cur(range.begin().GetValue());
    VID_T end = range.end().GetValue();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] =
          thread_pool_.enqueue([&cur, chunk_size, &iter_func, end, tid] {
            while (true) {
              VID_T cur_beg = std::min(cur.fetch_add(chunk_size), end);
              VID_T cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              VertexRange<VID_T> cur_range(cur_beg, cur_end);
              for (auto u : cur_range) {
                iter_func(tid, u);
              }
            }
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on discontinuous vertices concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param vertices The vertex array to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const VertexVector<VID_T>& vertices,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    std::atomic<size_t> cur(0);
    auto end = vertices.size();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue(
          [&cur, chunk_size, &vertices, &iter_func, end, tid] {
            while (true) {
              auto cur_beg = std::min(cur.fetch_add(chunk_size), end);
              auto cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              for (auto idx = cur_beg; idx < cur_end; idx++) {
                iter_func(tid, vertices[idx]);
              }
            }
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on vertexs of a VertexRange concurrently, initialize
   * function and finalize function can be provided to each thread.
   *
   * @tparam INIT_FUNC_T Type of thread init program.
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam FINALIZE_FUNC_T Type of thread finalize program.
   * @tparam VID_T Type of vertex id.
   * @param range The vertex range to be iterated.
   * @param init_func Initializing function to be invoked by each thread before
   * iterating on vertexs.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexs.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const VertexRange<VID_T>& range,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 1024) {
    std::atomic<VID_T> cur(range.begin().GetValue());
    VID_T end = range.end().GetValue();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue(
          [&cur, chunk_size, &init_func, &iter_func, &finalize_func, end, tid] {
            init_func(tid);

            while (true) {
              VID_T cur_beg = std::min(cur.fetch_add(chunk_size), end);
              VID_T cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              VertexRange<VID_T> cur_range(cur_beg, cur_end);
              for (auto u : cur_range) {
                iter_func(tid, u);
              }
            }

            finalize_func(tid);
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on discontinuous vertices concurrently, initialize
   * function and finalize function can be provided to each thread.
   *
   * @tparam INIT_FUNC_T Type of thread init program.
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam FINALIZE_FUNC_T Type of thread finalize program.
   * @tparam VID_T Type of vertex id.
   * @param vertices The vertex array to be iterated.
   * @param init_func Initializing function to be invoked by each thread before
   * iterating on vertexs.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexs.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const VertexVector<VID_T>& vertices,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 1024) {
    std::atomic<size_t> cur(0);
    auto end = vertices.size();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] =
          thread_pool_.enqueue([&cur, chunk_size, &init_func, &vertices,
                                &iter_func, &finalize_func, end, tid] {
            init_func(tid);

            while (true) {
              auto cur_beg = std::min(cur.fetch_add(chunk_size), end);
              auto cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              for (auto idx = cur_beg; idx < cur_end; idx++) {
                iter_func(tid, vertices[idx]);
              }
            }

            finalize_func(tid);
          });
    }

    thread_pool_.WaitEnd(results);
  }
  /**
   * @brief Iterate on vertexs of a DenseVertexSet concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VID_T>& dense_set,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    VertexRange<VID_T> range = dense_set.Range();
    std::atomic<VID_T> cur(range.begin().GetValue());
    VID_T beg = range.begin().GetValue();
    VID_T end = range.end().GetValue();

    const Bitset& bs = dense_set.GetBitset();
    chunk_size = ((chunk_size + 63) / 64) * 64;

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue(
          [&iter_func, &cur, chunk_size, &bs, beg, end, tid] {
            while (true) {
              VID_T cur_beg = std::min(cur.fetch_add(chunk_size), end);
              VID_T cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              for (VID_T vid = cur_beg; vid < cur_end; vid += 64) {
                Vertex<VID_T> v(vid);
                uint64_t word = bs.get_word(vid - beg);
                while (word != 0) {
                  if (word & 1) {
                    iter_func(tid, v);
                  }
                  ++v;
                  word = word >> 1;
                }
              }
            }
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on vertexs of a DenseVertexSet concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param range The vertex range to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VID_T>& dense_set,
                      const VertexRange<VID_T>& range,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    VertexRange<VID_T> complete_range = dense_set.Range();

    VID_T complete_begin = complete_range.begin().GetValue();
    VID_T complete_end = complete_range.end().GetValue();

    VID_T origin_begin = std::max(range.begin().GetValue(), complete_begin);
    VID_T origin_end = std::min(range.end().GetValue(), complete_end);

    VID_T batch_begin =
        (origin_begin - complete_begin + 63) / 64 * 64 + complete_begin;
    VID_T batch_end = (origin_end - complete_begin) / 64 * 64 + complete_begin;

    if (batch_begin >= origin_end || batch_end <= origin_begin) {
      Vertex<VID_T> v(origin_begin);
      Vertex<VID_T> end(origin_end);
      while (v != end) {
        if (dense_set.Exist(v)) {
          iter_func(0, v);
        }
        ++v;
      }
      return;
    }

    std::atomic<VID_T> cur(batch_begin);
    auto& bitset = dense_set.GetBitset();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue(
          [&iter_func, &cur, chunk_size, &bitset, batch_begin, batch_end,
           origin_begin, origin_end, complete_begin, this, tid] {
            if (tid == 0 && origin_begin < batch_begin) {
              Vertex<VID_T> v(origin_begin);
              Vertex<VID_T> end(batch_begin);
              while (v != end) {
                if (bitset.get_bit(v.GetValue())) {
                  iter_func(tid, v);
                }
                ++v;
              }
            }
            if (tid == (thread_num_ - 1) && batch_end < origin_end) {
              Vertex<VID_T> v(batch_end);
              Vertex<VID_T> end(origin_end);
              while (v != end) {
                if (bitset.get_bit(v.GetValue())) {
                  iter_func(tid, v);
                }
                ++v;
              }
            }
            if (batch_begin < batch_end) {
              while (true) {
                VID_T cur_beg = std::min(cur.fetch_add(chunk_size), batch_end);
                VID_T cur_end = std::min(cur_beg + chunk_size, batch_end);
                if (cur_beg == cur_end) {
                  break;
                }
                for (VID_T vid = cur_beg; vid < cur_end; vid += 64) {
                  Vertex<VID_T> v(vid);
                  uint64_t word = bitset.get_word(vid - complete_begin);
                  while (word != 0) {
                    if (word & 1) {
                      iter_func(tid, v);
                    }
                    ++v;
                    word = word >> 1;
                  }
                }
              }
            }
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on vertexs of a DenseVertexSet concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param vertices The vertices to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VID_T>& dense_set,
                      const VertexVector<VID_T>& vertices,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    std::atomic<size_t> cur(0);
    auto end = vertices.size();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] =
          thread_pool_.enqueue([&iter_func, &cur, chunk_size, &dense_set,
                                &vertices, end, this, tid] {
            while (true) {
              auto cur_beg = std::min(cur.fetch_add(chunk_size), end);
              auto cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              for (auto idx = cur_beg; idx < cur_end; idx++) {
                auto v = vertices[idx];
                if (dense_set.Exist(v)) {
                  iter_func(tid, v);
                }
              }
            }
          });
    }

    thread_pool_.WaitEnd(results);
  }
  /**
   * @brief Iterate on vertexs of a DenseVertexSet concurrently, initialize
   * function and finalize function can be provided to each thread.
   *
   * @tparam INIT_FUNC_T Type of thread init program.
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam FINALIZE_FUNC_T Type of thread finalize program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param init_func Initializing function to be invoked by each thread before
   * iterating on vertexs.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexs.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VID_T>& dense_set,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 10 * 1024) {
    VertexRange<VID_T> range = dense_set.Range();
    std::atomic<VID_T> cur(range.begin().GetValue());
    VID_T beg = range.begin().GetValue();
    VID_T end = range.end().GetValue();

    const Bitset& bs = dense_set.GetBitset();
    chunk_size = ((chunk_size + 63) / 64) * 64;

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] =
          thread_pool_.enqueue([&init_func, &finalize_func, &iter_func, &cur,
                                chunk_size, &bs, beg, end, tid] {
            init_func(tid);

            while (true) {
              VID_T cur_beg = std::min(cur.fetch_add(chunk_size), end);
              VID_T cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              for (VID_T vid = cur_beg; vid < cur_end; vid += 64) {
                Vertex<VID_T> v(vid);
                uint64_t word = bs.get_word(vid - beg);
                while (word != 0) {
                  if (word & 1) {
                    iter_func(tid, v);
                  }
                  ++v;
                  word = word >> 1;
                }
              }
            }

            finalize_func(tid);
          });
    }

    thread_pool_.WaitEnd(results);
  }

  uint32_t thread_num() { return thread_num_; }

 private:
  ThreadPool thread_pool_;
  uint32_t thread_num_;
};

template <typename APP_T>
typename std::enable_if<std::is_base_of<ParallelEngine, APP_T>::value>::type
InitParallelEngine(std::shared_ptr<APP_T> app, const ParallelEngineSpec& spec) {
  app->InitParallelEngine(spec);
}

template <typename APP_T>
typename std::enable_if<!std::is_base_of<ParallelEngine, APP_T>::value>::type
InitParallelEngine(std::shared_ptr<APP_T> app, const ParallelEngineSpec& spec) {
}

}  // namespace grape

#endif  // GRAPE_PARALLEL_PARALLEL_ENGINE_H_
