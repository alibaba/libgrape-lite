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
#include "grape/parallel/parallel_engine_spec.h"
#include "grape/utils/thread_pool.h"
#include "grape/utils/vertex_set.h"
#include "grape/worker/comm_spec.h"

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
   * @brief Iterate on vertexes of a VertexRange concurrently.
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
    auto dummy_func = [](int tid) {};
    ForEach(range, dummy_func, iter_func, dummy_func, chunk_size);
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DualVertexRange<VID_T>& range,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto dummy_func = [](int tid) {};
    ForEach(range, dummy_func, iter_func, dummy_func, chunk_size);
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
    auto dummy_func = [](int tid) {};
    ForEach(vertices, dummy_func, iter_func, dummy_func, chunk_size);
  }

  template <typename ITERATOR_T, typename ITER_FUNC_T>
  inline void ForEach(const ITERATOR_T& begin, const ITERATOR_T& end,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto dummy_func = [](int tid) {};
    ForEach(begin, end, dummy_func, iter_func, dummy_func, chunk_size);
  }

  /**
   * @brief Iterate on vertexes of a VertexRange concurrently, initialize
   * function and finalize function can be provided to each thread.
   *
   * @tparam INIT_FUNC_T Type of thread init program.
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam FINALIZE_FUNC_T Type of thread finalize program.
   * @tparam VID_T Type of vertex id.
   * @param range The vertex range to be iterated.
   * @param init_func Initializing function to be invoked by each thread before
   * iterating on vertexes.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexes.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const VertexRange<VID_T>& range,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 1024) {
    std::atomic<VID_T> cur(range.begin_value());
    VID_T end = range.end_value();

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

  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const DualVertexRange<VID_T>& range,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 1024) {
    VertexRange<VID_T> head = range.head();
    VertexRange<VID_T> tail = range.tail();
    std::atomic<VID_T> head_cur(head.begin_value());
    VID_T head_end = head.end_value();
    std::atomic<VID_T> tail_cur(tail.begin_value());
    VID_T tail_end = tail.end_value();

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue([&head_cur, &tail_cur, chunk_size,
                                           &iter_func, head_end, tail_end, tid,
                                           &init_func, &finalize_func] {
        init_func(tid);
        while (true) {
          VID_T cur_beg = std::min(head_cur.fetch_add(chunk_size), head_end);
          VID_T cur_end = std::min(cur_beg + chunk_size, head_end);
          if (cur_beg == cur_end) {
            break;
          }
          VertexRange<VID_T> cur_range(cur_beg, cur_end);
          for (auto& u : cur_range) {
            iter_func(tid, u);
          }
        }
        while (true) {
          VID_T cur_beg = std::min(tail_cur.fetch_add(chunk_size), tail_end);
          VID_T cur_end = std::min(cur_beg + chunk_size, tail_end);
          if (cur_beg == cur_end) {
            break;
          }
          VertexRange<VID_T> cur_range(cur_beg, cur_end);
          for (auto& u : cur_range) {
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
   * iterating on vertexes.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexes.
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
   * @brief Iterate a range specified by iterator pair concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam ITERATOR_T Type of range iterator.
   * @param begin The begin iterator of range.
   * @param end The end iterator of range.
   * @param init_func Initializing function to be invoked by each thread before
   * iterating on vertexes.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexes.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITERATOR_T, typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T>
  inline void ForEach(const ITERATOR_T& begin, const ITERATOR_T& end,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 1024) {
    std::atomic<size_t> offset(0);
    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] =
          thread_pool_.enqueue([&offset, chunk_size, &init_func, &iter_func,
                                &finalize_func, begin, end, tid] {
            init_func(tid);

            while (true) {
              const ITERATOR_T cur_beg =
                  std::min(begin + offset.fetch_add(chunk_size), end);
              const ITERATOR_T cur_end = std::min(cur_beg + chunk_size, end);
              if (cur_beg == cur_end) {
                break;
              }
              for (auto iter = cur_beg; iter != cur_end; ++iter) {
                iter_func(tid, *iter);
              }
            }

            finalize_func(tid);
          });
    }

    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on vertexes of a DenseVertexSet concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VertexRange<VID_T>>& dense_set,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto dummy_func = [](int tid) {};
    ForEach(dense_set, dummy_func, iter_func, dummy_func, chunk_size);
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VertexVector<VID_T>>& dense_set,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto dummy_func = [](int tid) {};
    ForEach(dense_set, dummy_func, iter_func, dummy_func, chunk_size);
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<DualVertexRange<VID_T>>& dense_set,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto dummy_func = [](int tid) {};
    ForEach(dense_set, dummy_func, iter_func, dummy_func, chunk_size);
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void bitwise_iterate(VID_T begin, VID_T end, const Bitset& bitset,
                              VID_T offset, int tid,
                              const ITER_FUNC_T& iter_func) {
    Vertex<VID_T> v(begin);
    Vertex<VID_T> v_end(end);
    while (v != v_end) {
      if (bitset.get_bit(v.GetValue() - offset)) {
        iter_func(tid, v);
      }
      ++v;
    }
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void wordwise_iterate(VID_T begin, VID_T end, const Bitset& bitset,
                               VID_T offset, int tid,
                               const ITER_FUNC_T& iter_func) {
    for (VID_T vid = begin; vid < end; vid += 64) {
      Vertex<VID_T> v(vid);
      uint64_t word = bitset.get_word(vid - offset);
      while (word != 0) {
        if (word & 1) {
          iter_func(tid, v);
        }
        ++v;
        word = word >> 1;
      }
    }
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void parallel_iterate(VID_T begin, VID_T end, const Bitset& bitset,
                               VID_T offset, const ITER_FUNC_T& iter_func,
                               int chunk_size) {
    VID_T batch_begin = (begin - offset + 63) / 64 * 64 + offset;
    VID_T batch_end = (end - offset) / 64 * 64 + offset;

    if (batch_begin >= end || batch_end <= begin) {
      bitwise_iterate(begin, end, bitset, offset, 0, iter_func);
      return;
    }

    std::atomic<VID_T> cur(batch_begin);
    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue([&iter_func, &cur, chunk_size,
                                           &bitset, batch_begin, batch_end,
                                           begin, end, offset, this, tid] {
        if (tid == 0 && begin < batch_begin) {
          bitwise_iterate(begin, batch_begin, bitset, offset, tid, iter_func);
        }
        if (tid == (thread_num_ - 1) && batch_end < end) {
          bitwise_iterate(batch_end, end, bitset, offset, tid, iter_func);
        }
        if (batch_begin < batch_end) {
          while (true) {
            VID_T cur_beg = std::min(cur.fetch_add(chunk_size), batch_end);
            VID_T cur_end = std::min(cur_beg + chunk_size, batch_end);
            if (cur_beg == cur_end) {
              break;
            }
            wordwise_iterate(cur_beg, cur_end, bitset, offset, tid, iter_func);
          }
        }
      });
    }
    thread_pool_.WaitEnd(results);
  }

  /**
   * @brief Iterate on vertexes of a DenseVertexSet concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param range The vertex range to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VertexRange<VID_T>>& dense_set,
                      const VertexRange<VID_T>& range,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto& bitset = dense_set.GetBitset();
    VertexRange<VID_T> complete_range = dense_set.Range();
    VID_T begin = std::max(range.begin_value(), complete_range.begin_value());
    VID_T end = std::min(range.end_value(), complete_range.end_value());
    if (begin < end) {
      parallel_iterate(begin, end, bitset, complete_range.begin_value(),
                       iter_func, chunk_size);
    }
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VertexVector<VID_T>>& dense_set,
                      const VertexRange<VID_T>& range,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    auto& bitset = dense_set.GetBitset();
    VertexRange<VID_T> complete_range = dense_set.Range();
    VID_T begin = std::max(range.begin_value(), complete_range.begin_value());
    VID_T end = std::min(range.end_value(), complete_range.end_value());
    if (begin < end) {
      parallel_iterate(begin, end, bitset, complete_range.begin_value(),
                       iter_func, chunk_size);
    }
  }

  template <typename ITER_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<DualVertexRange<VID_T>>& dense_set,
                      const VertexRange<VID_T>& range,
                      const ITER_FUNC_T& iter_func, int chunk_size = 1024) {
    VertexRange<VID_T> head_range = dense_set.Range().head();
    VertexRange<VID_T> tail_range = dense_set.Range().tail();

    VID_T head_begin = std::max(range.begin_value(), head_range.begin_value());
    VID_T head_end = std::min(range.end_value(), head_range.end_value());

    VID_T tail_begin = std::max(range.begin_value(), tail_range.begin_value());
    VID_T tail_end = std::min(range.end_value(), tail_range.end_value());

    auto& head_bitset = dense_set.GetHeadBitset();
    auto& tail_bitset = dense_set.GetTailBitset();

    if (head_begin < head_end) {
      parallel_iterate(head_begin, head_end, head_bitset,
                       head_range.begin_value(), iter_func, chunk_size);
    }
    if (tail_begin < tail_end) {
      parallel_iterate(tail_begin, tail_end, tail_bitset,
                       tail_range.begin_value(), iter_func, chunk_size);
    }
  }

  /**
   * @brief Iterate on vertexes of a DenseVertexSet concurrently.
   *
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param vertices The vertices to be iterated.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename ITER_FUNC_T, typename VID_T, typename VERTEX_SET_T>
  inline void ForEach(const DenseVertexSet<VERTEX_SET_T>& dense_set,
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
   * @brief Iterate on vertexes of a DenseVertexSet concurrently, initialize
   * function and finalize function can be provided to each thread.
   *
   * @tparam INIT_FUNC_T Type of thread init program.
   * @tparam ITER_FUNC_T Type of vertex program.
   * @tparam FINALIZE_FUNC_T Type of thread finalize program.
   * @tparam VID_T Type of vertex id.
   * @param dense_set The vertex set to be iterated.
   * @param init_func Initializing function to be invoked by each thread before
   * iterating on vertexes.
   * @param iter_func Vertex program to be applied on each vertex.
   * @param finalize_func Finalizing function to be invoked by each thread after
   * iterating on vertexes.
   * @param chunk_size Vertices granularity to be scheduled by threads.
   */
  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VertexRange<VID_T>>& dense_set,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 10 * 1024) {
    VertexRange<VID_T> range = dense_set.Range();
    std::atomic<VID_T> cur(range.begin_value());
    VID_T beg = range.begin_value();
    VID_T end = range.end_value();

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

  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<VertexVector<VID_T>>& dense_set,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 10 * 1024) {
    VertexRange<VID_T> range = dense_set.Range();
    std::atomic<VID_T> cur(range.begin_value());
    VID_T beg = range.begin_value();
    VID_T end = range.end_value();

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

  template <typename INIT_FUNC_T, typename ITER_FUNC_T,
            typename FINALIZE_FUNC_T, typename VID_T>
  inline void ForEach(const DenseVertexSet<DualVertexRange<VID_T>>& dense_set,
                      const INIT_FUNC_T& init_func,
                      const ITER_FUNC_T& iter_func,
                      const FINALIZE_FUNC_T& finalize_func,
                      int chunk_size = 10 * 1024) {
    VertexRange<VID_T> head = dense_set.head();
    VertexRange<VID_T> tail = dense_set.tail();
    VID_T head_beg = head.begin_value();
    std::atomic<VID_T> head_cur(head_beg);
    VID_T head_end = head.end_value();
    VID_T tail_beg = tail.begin_value();
    std::atomic<VID_T> tail_cur(tail_beg);
    VID_T tail_end = tail.end_value();

    const Bitset& head_bs = dense_set.GetHeadBitset();
    const Bitset& tail_bs = dense_set.GetTailBitset();
    chunk_size = ((chunk_size + 63) / 64) * 64;

    std::vector<std::future<void>> results(thread_num_);
    for (uint32_t tid = 0; tid < thread_num_; ++tid) {
      results[tid] = thread_pool_.enqueue([&init_func, &finalize_func,
                                           &iter_func, &head_cur, &tail_cur,
                                           chunk_size, &head_bs, &tail_bs,
                                           head_beg, tail_beg, head_end,
                                           tail_end, tid] {
        init_func(tid);
        while (true) {
          VID_T cur_beg = std::min(head_cur.fetch_add(chunk_size), head_end);
          VID_T cur_end = std::min(cur_beg + chunk_size, head_end);
          if (cur_beg == cur_end) {
            break;
          }
          for (VID_T vid = cur_beg; vid < cur_end; vid += 64) {
            Vertex<VID_T> v(vid);
            uint64_t word = head_bs.get_word(vid - head_beg);
            while (word != 0) {
              if (word & 1) {
                iter_func(tid, v);
              }
              ++v;
              word = word >> 1;
            }
          }
        }
        while (true) {
          VID_T cur_beg = std::min(tail_cur.fetch_add(chunk_size), tail_end);
          VID_T cur_end = std::min(cur_beg + chunk_size, tail_end);
          if (cur_beg == cur_end) {
            break;
          }
          for (VID_T vid = cur_beg; vid < cur_end; vid += 64) {
            Vertex<VID_T> v(vid);
            uint64_t word = tail_bs.get_word(vid - tail_beg);
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
