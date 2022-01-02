#ifndef GRAPEGPU_GRAPE_GPU_UTILS_PC_QUEUE_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_PC_QUEUE_H_
#include <cooperative_groups.h>
#include <thrust/device_vector.h>

#include <map>
#include <vector>

#include "glog/logging.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/launcher.h"
#include "grape_gpu/utils/segment.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/array_view.h"

namespace grape_gpu {
namespace dev {
/*
 * @brief A device-level Producer-Consumer Queue (see host controller object
 * below for usage and notes)
 */
template <typename T>
class PCQueue {
  T* data_{};
  uint32_t *start_{}, *end_{}, *pending_{};
  uint32_t capacity_mask_{};

 public:
  DEV_HOST PCQueue() {}
  DEV_HOST PCQueue(T* data, uint32_t* start, uint32_t* end, uint32_t* pending,
                   uint32_t capacity)
      : data_(data),
        start_(start),
        end_(end),
        pending_(pending),
        capacity_mask_((capacity - 1)) {
    // Must be a power of two for handling circular overflow correctly
    assert((capacity - 1 & capacity) == 0);
  }

  DEV_HOST PCQueue(const PCQueue& rhs) : data_(rhs.data_), start_(rhs.start_),
  end_(rhs.end_) , pending_(rhs.pending_), capacity_mask_(rhs.capacity_mask_) {}

  DEV_INLINE void Reset() const {
    *start_ = 0;
    *end_ = 0;
    *pending_ = 0;
  }

  DEV_INLINE void Pop(uint32_t count) const { (*start_) += count; }

  //FIXME overflow
  DEV_INLINE void Appendv(const T* item, size_t size) {
    uint32_t allocation = atomicAdd((uint32_t*) pending_, size);
    for(uint32_t pos = 0; pos < size; ++pos) {
      uint32_t _pos = allocation + pos;
      data_[_pos & capacity_mask_] = item[pos];
    }
  }

  //FIXME overflow
  DEV_INLINE void AppendWarpv(const T* item, size_t size) {
    auto g = cooperative_groups::coalesced_threads();
    uint32_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicAdd((uint32_t*) pending_, g.size() * size);
    }

    auto allocation = g.shfl(warp_res, 0) + g.thread_rank() * size;
    for(uint32_t pos = 0; pos < size; pos++) {
      uint32_t _pos = allocation + size;
      data_[_pos & capacity_mask_] = item[pos];
    }
  }

  DEV_INLINE bool ReadAndPopv (T* item, size_t size) {
    if(Count() < size) return false;
    uint32_t allocation = atomicAdd((uint32_t*) start_, size);
    for(uint32_t pos = 0; pos < size; ++pos) {
      uint32_t _pos = allocation + pos;
      item[pos] = data_[_pos & capacity_mask_];
    }
    return true;
  }

  DEV_INLINE bool ReadAndPopWarpv (T* item, size_t size) {
    auto g = cooperative_groups::coalesced_threads();
    if(Count() < g.size() * size) return false;
    uint32_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicAdd((uint32_t*) start_, g.size() * size);
    }

    auto allocation = g.shfl(warp_res, 0) + g.thread_rank() * size;
    for(uint32_t pos = 0; pos < size; pos++) {
      uint32_t _pos = allocation + size;
      item[pos] = data_[_pos & capacity_mask_];
    }
    return true;
  }
  
  DEV_INLINE void Append(const T& item) {
    uint32_t allocation = atomicAdd((uint32_t*) pending_, 1);
    data_[allocation & capacity_mask_] = item;
  }

  DEV_INLINE void AppendWarp(const T& item) {
    auto g = cooperative_groups::coalesced_threads();
    uint32_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicAdd((uint32_t*) pending_, g.size());
    }
    auto allocation = g.shfl(warp_res, 0) + g.thread_rank();
    data_[allocation & capacity_mask_] = item;
  }

  DEV_INLINE void Prepend(const T& item) {
    uint32_t allocation = atomicSub((uint32_t*) start_, 1) - 1;
    data_[allocation & capacity_mask_] = item;
  }

  DEV_INLINE void PrependWarp(const T& item) {
    auto g = cooperative_groups::coalesced_threads();
    uint32_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicSub((uint32_t*) start_, g.size()) - g.size();
      assert(*end_ - warp_res < (capacity_mask_ + 1));
    }
    auto allocation = g.shfl(warp_res, 0) + g.thread_rank();
    data_[allocation & capacity_mask_] = item;
  }

  DEV_INLINE T Read(uint32_t i) const {
    return data_[(*start_ + i) & capacity_mask_];
  }

  DEV_INLINE uint32_t Count() const { return *end_ - *start_; }

  // Returns the 'count' of pending items and commits
  DEV_INLINE uint32_t CommitPending() const {
    uint32_t count = *pending_ - *end_;

    // Sync end with pending, this makes the pushed items visible to the
    // consumer
    *end_ = *pending_;
    return count;
  }
};
}  // namespace dev

class QueueMemoryMonitor {
  struct Entry {
    int device_id;
    const char* name;
    size_t capacity;
    size_t max_usage;

    Entry() : device_id(), name(nullptr), capacity(0), max_usage(0) {}
    Entry(int device_id, const char* name, size_t capacity)
        : device_id(device_id), name(name), capacity(capacity), max_usage(0) {}
  };

  int curr_index_;
  std::map<int, Entry> entries_;  // <index, Entry>
  std::atomic<bool> exiting_;

  QueueMemoryMonitor() : curr_index_(0), exiting_(false) {}

  static QueueMemoryMonitor& Instance() {
    static QueueMemoryMonitor monitor;  // Lazy singleton (per compilation unit)
    return monitor;
  }

  int RegisterInternal(size_t capacity, int device_id, const char* name) {
    int index = curr_index_++;
    entries_[index] = Entry(device_id, name, capacity);
    return index;
  }

  void UnregisterInternal(int index) {
    if (entries_.find(index) != entries_.end()) {
      entries_.erase(entries_.find(index));
    }
  }

  void ReportUsageInternal(int index, size_t usage) {
    if (entries_.find(index) == entries_.end()) {
      LOG(FATAL) << "Unregistered PCQueue " << index;
    }
    auto& curr_entry = entries_.at(index);

    curr_entry.max_usage = std::max(curr_entry.max_usage, usage);

    if (usage > curr_entry.capacity) {
      //
      // We got an overflow, report overall usage stats and exit
      //
      bool exiting = exiting_;

      if (exiting || !exiting_.compare_exchange_strong(exiting, true)) {
        return;  // Avoid printing stats by more than one thread
      }
      std::map<int, std::map<std::string, std::vector<Entry>>> grouped_entries;

      // Group by device_id and name
      for (const auto& ep : entries_) {
        auto& entry = ep.second;

        grouped_entries[entry.device_id][entry.name].push_back(entry);
      }

      std::stringstream ss;
      ss << "Queue has overflowed, dumping overall queue memory "
            "statistics:\n";

      for (const auto& ep : grouped_entries) {
        auto device_id = ep.first;

        ss << "Device " << device_id << ":\n";

        for (const auto& np : ep.second) {
          std::string name = np.first;

          for (auto grouped_entry : np.second) {
            ss << "\tname: " << name << ", capacity: " << grouped_entry.capacity
               << ", max usage: " << grouped_entry.max_usage
               << ", overflowed ratio: "
               << (double) grouped_entry.max_usage / grouped_entry.capacity
               << ", overflowed count: "
               << grouped_entry.max_usage - grouped_entry.capacity << "\n";
          }
        }
      }
      LOG(FATAL) << ss.str();
    }
  }

 public:
  static int Register(size_t capacity, int device_id, const char* name) {
    return Instance().RegisterInternal(capacity, device_id, name);
  }

  static void Unregister(int index) {
    return Instance().UnregisterInternal(index);
  }

  static void ReportUsage(int instance_id, size_t usage) {
    Instance().ReportUsageInternal(instance_id, usage);
  }
};

/*
* @brief Host controller object for dev::PCQueue (see above)
* @notes:
    -> The producer-consumer queue supports concurrent access from two
kernels/streams running over the same device
    -> All threads in the consumer kernel can read items from the queue, but
only a single thread pop's after all others are done
    -> All threads in the producer kernel can push values into the queue, but
only a single thread commit's at the end, signaling to readers
    -> The consumer kernel can also have a write phase where it 'prepends' items
into the queue
    -> In order to support a lock-free implementation (with only atomics), the
capacity of the queue must be of power-of-two so that numeric under/over-flows
are correctly managed allong with the modulo test
    -> Power-of-two capacity enables the use of a capacity mask for the modulo
test
    -> The queue does not support blocking writers when it's full, instead, a
memory overflow is reported and the application will terminate
*/
template <typename T>
class PCQueue {
  using SIZE_T = uint32_t;
  using DeviceObjectType = dev::PCQueue<T>;

 public:
  PCQueue() = default;

  explicit PCQueue(SIZE_T capacity, const char* name = "") {
    Init(capacity, name);
  }

  ~PCQueue() { QueueMemoryMonitor::Unregister(instance_id_); }

  void Init(SIZE_T capacity, const char* name = "") {
    int device_id;

    CHECK_CUDA(cudaGetDevice(&device_id));

    if ((capacity - 1 & capacity) != 0) {
      int power = 1;
      while (power < capacity) {
        power *= 2;
      }
      capacity = power;
      VLOG(1) << "Init queue " << name << " on device " << device_id
              << " with capacity: " << capacity;
    }

    start_.set(0);
    end_.set(0);
    pending_.set(0);

    data_.resize(capacity);
    instance_id_ = QueueMemoryMonitor::Register(capacity, device_id, name);
  }

  DeviceObjectType _DeviceObject() {
    return dev::PCQueue<T>(thrust::raw_pointer_cast(data_.data()),
                           start_.data(), end_.data(), pending_.data(),
                           data_.size());
  }

  uint32_t GetCapacity() {
    return this->data_.size();
  }

  T* Data() {
    return thrust::raw_pointer_cast(this->data_.data());
  }

  void ResetAsync(const Stream& stream) {
    LaunchKernel(
        stream,
        [] __device__(DeviceObjectType pc_queue) {
          if (TID_1D == 0) {
            pc_queue.Reset();
          }
        },
        _DeviceObject());
  }

  void CommitPendingAsync(const Stream& stream) {
    LaunchKernel(
        stream,
        [] __device__(DeviceObjectType pc_queue) {
          if (TID_1D == 0) {
            pc_queue.CommitPending();
          }
        },
        _DeviceObject());
  }

  void AppendItemAsync(const Stream& stream, const T& item) {
    LaunchKernel(
        stream,
        [item] __device__(DeviceObjectType pc_queue) {
          if (TID_1D == 0) {
            pc_queue.append(item);
            pc_queue.commit_pending();
          }
        },
        _DeviceObject());
  }

  void AppendItems(const Stream& stream, ArrayView<T> array) {
    LaunchKernel(
        stream,
        [array] __device__(DeviceObjectType pc_queue) {
          for (size_t idx = TID_1D; idx < array.size();
               idx += TOTAL_THREADS_1D) {
            pc_queue.AppendWarp(array[idx]);
          }
        },
        _DeviceObject());
  }

  void PopAsync(const Stream& stream, SIZE_T count) {
    if (count != 0) {
      LaunchKernel(
          stream,
          [count] __device__(DeviceObjectType pc_queue) {
            if (TID_1D == 0) {
              pc_queue.Pop(count);
            }
          },
          _DeviceObject());
    }
  }

  void PopAsync(const Stream& stream) {
    CHECK_CUDA(cudaMemcpyAsync(start_.data(), end_.data(), sizeof(SIZE_T),
                               cudaMemcpyDefault, stream.cuda_stream()));
  }

  SIZE_T GetCount(const Stream& stream) const {
    SIZE_T start, end, size;
    GetBounds(start, end, size, stream);

    return size;
  }

  SIZE_T GetPendingCount(const Stream& stream) const {
    SIZE_T end, pending, count;
    GetPendingCount(end, pending, count, stream);

    return count;
  }

  SIZE_T GetSpace(const Stream& stream) const {
    return data_.size() - GetCount(stream);
  }

  
  void PourSeg(Segment<T>& seg, const Stream& stream) {
    SIZE_T start, pending, count, capacity;
    GetRemainCount(start, pending, count, stream);
    assert(seg.size() <= count);
    capacity = data_.size();
    T* src_ptr = seg.GetSegmentptr();
    T* dst_ptr = (T*)thrust::raw_pointer_cast(data_.data()) + pending;
    SIZE_T cpy_size = seg.size();
    SIZE_T limit = pending > start ? (capacity-pending) : (start-pending);
    if(pending > start && limit < cpy_size){ //two memcpy
      cpy_size -= limit;
      CHECK_CUDA(cudaMemcpyAsync((void*)dst_ptr, (void*)src_ptr,
          limit * sizeof(T), cudaMemcpyDeviceToDevice, stream.cuda_stream()));
      src_ptr += limit;
      dst_ptr = (T*)thrust::raw_pointer_cast(data_.data());
    }
    if(cpy_size) {
      CHECK_CUDA(cudaMemcpyAsync((void*)dst_ptr, (void*)src_ptr,
          cpy_size * sizeof(T), cudaMemcpyDeviceToDevice, stream.cuda_stream()));
    }
  }

  std::vector<Segment<T>> GetSegs(const Stream& stream) {
    return GetSegs(GetBounds(stream));
  }

  //
  // Debug methods
  //
  void GetOffsets(SIZE_T& capacity, SIZE_T& start, SIZE_T& end, SIZE_T& pending,
                  SIZE_T& size, const Stream& stream) const {
    capacity = data_.size();
    start = start_.get(stream);
    end = end_.get(stream);
    pending = pending_.get(stream);
    size = end - start;
  }

  bool Empty(const Stream& stream ) const{
    SIZE_T start, end, pending, size, capacity;
    GetOffsets(capacity, start, end, pending, size, stream);
    if (start == end && end == pending) return true;
    else return false;
  }

  void PrintOffsets(const Stream& stream) const {
    SIZE_T capacity, start, end, pending, size;

    GetOffsets(capacity, start, end, pending, size, stream);
    LOG(INFO) << "PCQueue (Debug): start: " << start << ", end: " << end
              << ", pending: " << pending << ", size: " << size
              << " (capacity: " << capacity << ")";
  }

 private:
  // Device buffer / counters
  thrust::device_vector<T> data_;
  SharedValue<SIZE_T> start_, end_, pending_;
  int instance_id_;

  //
  // Helper class for queue bounds
  //
  struct Bounds {
    SIZE_T start, end;

    Bounds(SIZE_T start, SIZE_T end) : start(start), end(end) {}
    Bounds() : start(0), end(0) {}

    SIZE_T GetCount() const {
      return end - start;
    }  // Works also if numbers over/under flow
    Bounds Exclude(Bounds other) const { return Bounds(other.end, end); }
  };

  Bounds GetBounds(const Stream& stream) {
    Bounds bounds;
    GetActualBounds(bounds.start, bounds.end, stream);
    return bounds;
  }

  SIZE_T GetSpace(Bounds bounds) const {
    return data_.size() - bounds.GetCount();
  }

  std::vector<Segment<T>> GetSegs(Bounds bounds) {
    auto start = bounds.start, end = bounds.end, size = bounds.GetCount();
    auto capacity = data_.size();

    start = start % capacity;
    end = end % capacity;

    std::vector<Segment<T>> segs;

    if (end > start) {  // normal case
      segs.push_back(
          Segment<T>(thrust::raw_pointer_cast(data_.data()) + start, size));
    } else if (start > end) {
      segs.push_back(Segment<T>(thrust::raw_pointer_cast(data_.data()) + start,
                                size - end));
      if (end > 0) {
        segs.push_back(Segment<T>(thrust::raw_pointer_cast(data_.data()), end));
      }
    }

    // else empty
    return segs;
  }

  void GetActualBounds(SIZE_T& start, SIZE_T& end, const Stream& stream) const {
    start = start_.get(stream);
    end = end_.get(stream);

    QueueMemoryMonitor::ReportUsage(instance_id_, end - start);
  }

  void GetBounds(SIZE_T& start, SIZE_T& end, SIZE_T& size,
                 const Stream& stream) const {
    auto capacity = data_.size();

    GetActualBounds(start, end, stream);

    start = start % capacity;
    end = end % capacity;

    size = end >= start
               ? end - start
               : (capacity - start + end);  // normal and circular cases
  }

  void GetPendingCount(SIZE_T& end, SIZE_T& pending, SIZE_T& count,
                       const Stream& stream) const {
    auto capacity = data_.size();

    end = end_.get(stream);
    pending = pending_.get(stream);

    QueueMemoryMonitor::ReportUsage(instance_id_, pending - end);

    end = end % capacity;
    pending = pending % capacity;

    count = pending >= end
                ? pending - end
                : (capacity - end + pending);  // normal and circular cases
  }

  void GetRemainCount(SIZE_T& start, SIZE_T& pending, SIZE_T& count, 
       const Stream& stream) const {
    start = start_.get(stream);
    pending = pending_.get(stream);

    count = start - pending;
  }
};

}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_PC_QUEUE_H_
