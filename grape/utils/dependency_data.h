
#ifndef AUTOINC_GRAPE_UTILS_DEPENDENCY_DATA_H_
#define AUTOINC_GRAPE_UTILS_DEPENDENCY_DATA_H_
#include "grape/graph/vertex.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/utils/atomic_ops.h"

namespace grape {

template <typename VID_T, typename T>
struct DependencyData {
  static_assert(std::is_pod<T>::value, "Unsupported type");
  VID_T parent_gid;
  T value;

  DependencyData() : parent_gid(std::numeric_limits<VID_T>::max()) {}

  DependencyData(VID_T p, T val) : parent_gid(p), value(val) {}

  inline bool SetIfLessThan(const DependencyData<VID_T, T>& rhs) {
    DependencyData<VID_T, T> old_delta;
    // auto& new_delta = const_cast<DependencyData<VID_T, T>&>(rhs);
    auto new_delta = rhs;

    do {
      old_delta = *this;

      if (old_delta.value <= new_delta.value) {
        return false;
      }
    } while (!CAS(this, old_delta, new_delta));
    return true;
  }

  inline bool SetIfGreaterThan(const DependencyData<VID_T, T>& rhs) {
    DependencyData<VID_T, T> old_delta;
    auto& new_delta = const_cast<DependencyData<VID_T, T>&>(rhs);

    do {
      old_delta = *this;

      if (old_delta.value >= new_delta.value) {
        return false;
      }
    } while (!CAS(this, old_delta, new_delta));
    return true;
  }

  friend InArchive& operator<<(InArchive& archive,
                               const DependencyData<VID_T, T>& h) {
    archive << h.parent_gid;
    archive << h.value;
    return archive;
  }

  friend OutArchive& operator>>(OutArchive& archive,
                                DependencyData<VID_T, T>& h) {
    archive >> h.parent_gid;
    archive >> h.value;
    return archive;
  }

  void Reset(T val) {
    parent_gid = std::numeric_limits<VID_T>::max();
    value = val;
  }
};

}  // namespace grape
#endif  // AUTOINC_GRAPE_UTILS_DEPENDENCY_DATA_H_
