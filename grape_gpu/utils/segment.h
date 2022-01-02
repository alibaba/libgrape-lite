
#ifndef GRAPEGPU_GRAPE_GPU_UTILS_SEGMENT_H_
#define GRAPEGPU_GRAPE_GPU_UTILS_SEGMENT_H_
#include "glog/logging.h"
namespace grape_gpu {
/*
 * @brief Represents a segment copy of valid data (for some known datum)
 */
template <typename T>
class Segment {
 private:
  T* segment_ptr_;
  size_t total_size_;
  size_t segment_size_;
  size_t segment_offset_;

 public:
  Segment(T* segment_ptr, size_t total_size, size_t segment_size,
          size_t segment_offset, void* metadata = nullptr)
      : segment_ptr_(segment_ptr),
        total_size_(total_size),
        segment_size_(segment_size),
        segment_offset_(segment_offset),
        metadata(metadata) {}

  Segment(T* segment_ptr, size_t total_size, void* metadata = nullptr)
      : segment_ptr_(segment_ptr),
        total_size_(total_size),
        segment_size_(total_size),
        segment_offset_(0),
        metadata(metadata) {}

  Segment()
      : segment_ptr_(nullptr),
        total_size_(0),
        segment_size_(0),
        segment_offset_(0),
        metadata(nullptr) {}

  void* metadata;  // a metadata field for user customization

  /// @brief is the segment empty
  bool Empty() const { return segment_size_ == 0; }

  /// @brief a pointer to segment start
  T* GetSegmentPtr() const { return segment_ptr_; }

  /// @brief The total size of the source datum
  size_t GetTotalSize() const { return total_size_; }

  /// @brief The size of the segment
  size_t GetSegmentSize() const { return segment_size_; }

  /// @brief The offset within the original buffer
  size_t GetSegmentOffset() const { return segment_offset_; }

  /// @brief Builds a sub-segment out of this segment
  Segment<T> GetSubSegment(size_t relative_offset,
                           size_t sub_segment_size) const {
    if (relative_offset > segment_size_ ||
        sub_segment_size > segment_size_ - relative_offset) {
      LOG(FATAL) << "Out of range";  // out of segment range
    }
    return Segment<T>(segment_ptr_ + relative_offset, total_size_,
                      sub_segment_size, segment_offset_ + relative_offset,
                      metadata);
  }

  Segment<T> GetFirstSubSegment(size_t max_subseg_size) const {
    if (max_subseg_size == 0)
      return {*this};

    return GetSubSegment(
        0, (size_t) ((max_subseg_size) > segment_size_ ? (segment_size_)
                                                       : max_subseg_size));
  }

  std::vector<Segment<T>> Split(size_t max_subseg_size) const {
    if (max_subseg_size == 0)
      return {*this};

    std::vector<Segment<T>> subsegs;
    subsegs.reserve(round_up(segment_size_, max_subseg_size));

    size_t pos = 0;

    while (pos < segment_size_) {
      subsegs.push_back(
          GetSubSegment(pos, (size_t) ((pos + max_subseg_size) > segment_size_
                                           ? (segment_size_ - pos)
                                           : max_subseg_size)));

      pos += max_subseg_size;
    }

    return subsegs;
  }
};
}  // namespace grape_gpu
#endif  // GRAPEGPU_GRAPE_GPU_UTILS_SEGMENT_H_
