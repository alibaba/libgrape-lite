#ifndef GRAPE_GPU_SERIALIZATION_ASYNC_IN_ARCHIVE_H_
#define GRAPE_GPU_SERIALIZATION_ASYNC_IN_ARCHIVE_H_
#include <thrust/device_vector.h>
#include <thrust/pair.h>

#include <cub/cub.cuh>

#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/shared_value.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/pc_queue.h"

namespace grape_gpu {
namespace dev {

template<typename S>
class AsyncInArchive : public dev::PCQueue<S> {
 public:
  DEV_HOST AsyncInArchive() : PCQueue<S>() {}

  DEV_HOST AsyncInArchive(S* data, uint32_t* start, uint32_t* end, 
                 uint32_t* pending, uint32_t capa)
    : PCQueue<S>(data, start, end, capa) {}

  DEV_HOST AsyncInArchive(const PCQueue<S>& q) : dev::PCQueue<S>(q){}

  template <typename T>
  DEV_INLINE void AddBytes(T elem) {
    //assert(sizeof(T)/sizeof(S)*sizeof(S)!=sizeof(T));
    Appendv(reinterpret_cast<S*>(&elem), sizeof(T)/sizeof(S));
  }

  template <typename T>
  DEV_INLINE void AddBytesWarp(T elem) {
    AppendWarpv(reinterpret_cast<S*>(&elem), sizeof(T)/sizeof(S));
  }
};

}  // namespace dev

template<typename S>
class AsyncInArchive : public PCQueue<S> {
 public:
  AsyncInArchive() : PCQueue<S>(0) {}

  explicit AsyncInArchive(uint32_t capacity) : PCQueue<S>(capacity) {}

  dev::AsyncInArchive<S> DeviceObject() { // comp_thread
    return dev::AsyncInArchive<S>(this->_DeviceObject());
  }

  uint32_t capacity() const { return this->GetCapacity(); }

  void Commit(const Stream& stream) {  //comp_thread
    this->CommitPendingAsync(stream);
  }

  size_t GetCount(const Stream& stream){ // comm_thread
    return this->GetCount(stream);
  }

  void PopAsync(const Stream& stream, size_t count) { // comm_thread
    this->PopAsync(stream, count);
  }

  // TODO return S but need T (done)
  // TODO handle the truncated data due to the circle queue.
  template<typename T>
  std::vector<Segment<T>> FetchSegs(const Stream& stream) { // comm_thread
    std::vector<Segment<S>> Ssegs = this->GetSegs(stream);
    std::vector<Segment<T>> Tsegs;
    for(size_t i = 0; i < Ssegs.size(); ++i) {
      Tsegs.emplace_back(reinterpret_cast<T*>(Ssegs[i].GetSegmentPtr()),
                            Ssegs[i].GetTotalSize() * sizeof(S)/ sizeof(T));
    }
    return Tsegs; // RVO?
  }

  template<typename T>
  T* data() { return reinterpret_cast<T*>(this->Data()); }

  //void Allocate(uint32_t capacity) {}
  
  //uint32_t size() const { return size_.get(); }

  //uint32_t size(const Stream& stream) const { return size_.get(stream); }

  //void Clear() { size_.set(0); }

  //void Clear(const Stream& stream) { size_.set(0, stream); }

  //bool Empty() const { return size_.get() == 0; }

  //bool Empty(const Stream& stream) const { return size_.get(stream) == 0; }

};
}  // namespace grape_gpu

#endif  // GRAPE_GPU_SERIALIZATION_ASYNC_IN_ARCHIVE_H_
