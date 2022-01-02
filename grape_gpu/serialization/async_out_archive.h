#ifndef GRAPE_GPU_SERIALIZATION_ASYNC_OUT_ARCHIVE_H_
#define GRAPE_GPU_SERIALIZATION_ASYNC_OUT_ARCHIVE_H_
#include <cooperative_groups.h>

#include "grape/util.h"
#include "grape_gpu/utils/array_view.h"
#include "grape_gpu/utils/cuda_utils.h"
#include "grape_gpu/utils/dev_utils.h"
#include "grape_gpu/utils/stream.h"
#include "grape_gpu/utils/pc_queue.h"

namespace grape_gpu {
namespace dev {

template<typename S>
class AsyncOutArchive : public dev::PCQueue<S> {
 public:
  DEV_HOST AsyncOutArchive() : PCQueue<S>() {}

  DEV_HOST AsyncOutArchive(S* data, uint32_t* start, uint32_t* end,
                  uint32_t* pending, uint32_t capa)
      : PCQueue<S>(data, start, end, capa) {}

  DEV_HOST AsyncOutArchive(const PCQueue<S>& q) : dev::PCQueue<S>(q){}

  template <typename T>
  DEV_INLINE bool GetBytes(T& elem) {
    //assert(sizeof(T)/sizeof(S)*sizeof(S)!=sizeof(T));
    return this->ReadAndPopv(reinterpret_cast<S*>(&elem), sizeof(T)/sizeof(S)); 
  }

  template <typename T>
  DEV_INLINE bool GetBytesWarp(T& elem) {
    return this->ReadAndPopWarpv(reinterpret_cast<S*>(&elem), sizeof(T)/sizeof(S)); 
  }

  DEV_INLINE bool Empty() const {
    return 0 == this->Count();
  }

  DEV_INLINE uint32_t size() const { 
    return this->Count(); 
  }
};

}  // namespace dev

template<typename S>
class AsyncOutArchive : public PCQueue<S>{
 public:
  AsyncOutArchive() : AsyncOutArchive(0) {}

  explicit AsyncOutArchive(uint32_t capacity) : PCQueue<S>(capacity) {}

  dev::AsyncOutArchive<S> DeviceObject() { // comm_thread
    return dev::AsyncOutArchive<S>(this->_DeviceObject());
  }

  template<typename T>
  void PourSeg(Segment<T>& seg, const Stream& stream) {
    Segment<S> Sseg(reinterpret_cast<S*>(seg.GetSegmentPtr()), 
                    seg.GetTotalSize() * sizeof(S) / sizeof(T));
    this->PourSeg(Sseg, stream); 
  }

  uint32_t capacity() const { return this->GetCapacity(); }

  void Commit(const Stream& stream) {  //com,_thread
    this->CommitPendingAsync(stream);
  }

  size_t GetCount(const Stream& stream){ // comp_thread
    return this->GetCount(stream);
  }

  void PopAsync(const Stream& stream, size_t count) { // comp_thread
    this->PopAsync(stream, count);
  }

  // TODO return S but need T (done)
  // TODO handle the truncated data due to the circle queue.
  template<typename T>
  std::vector<Segment<T>> FetchSegs(const Stream& stream) { // comp_thread
    std::vector<Segment<S>> Ssegs = this->GetSegs(stream);
    std::vector<Segment<T>> Tsegs;
    for(size_t i = 0; i < Ssegs.size(); ++i) {
      Tsegs.emplace_back(reinterpret_cast<T*>(Ssegs[i].GetSegmentPtr()), 
                            Ssegs[i].GetTotalSize() * sizeof(S) / sizeof(T));
    }
    return Tsegs; // RVO?
  }

  template<typename T>
  T* data() { return reinterpret_cast<T*>(this->Data()); }

  //void Allocate(uint32_t capacity) { buffer_.resize(capacity); }

  //void Clear() {
  //  limit_ = 0;
  //  pos_.set(0);
  //}

  //void Clear(const Stream& stream) {
  //  limit_ = 0;
  //  pos_.set(0, stream);
  //}


  //char* data() { return thrust::raw_pointer_cast(buffer_.data()); }

  //void SetLimit(uint32_t limit) { limit_ = limit; }

  //uint32_t AvailableBytes() const { return limit_ - pos_.get(); }

  //uint32_t AvailableBytes(const Stream& stream) const {
  //  return limit_ - pos_.get(stream);
  //}
};
}  // namespace grape_gpu

#endif  // GRAPE_GPU_SERIALIZATION_ASYNC_OUT_ARCHIVE_H_
