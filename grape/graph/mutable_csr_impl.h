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

#ifndef GRAPE_GRAPH_MUTABLE_CSR_IMPL_H_
#define GRAPE_GRAPH_MUTABLE_CSR_IMPL_H_

#include <algorithm>

#include "grape/config.h"
#include "grape/graph/adj_list.h"

namespace grape {

namespace mutable_csr_impl {

template <typename NBR_T>
struct AdjList {
  AdjList() : begin(NULL), end(NULL) {}

  int degree() const { return (end - begin); }

  bool empty() const { return (begin == end); }

  NBR_T* begin;
  NBR_T* end;
};

template <typename VID_T, typename T>
class Blob {
 public:
  Blob() {}
  explicit Blob(size_t size) : buffer_(size) {}
  ~Blob() {}

  Blob(Blob&& rhs) : buffer_(std::move(rhs.buffer_)) {}

  void resize(size_t size) { buffer_.resize(size); }

  T* data() { return buffer_.data(); }
  const T* data() const { return buffer_.data(); }

  T& operator[](VID_T index) { return buffer_[index]; }
  const T& operator[](VID_T index) const { return buffer_[index]; }

  size_t size() const { return buffer_.size(); }

 private:
  Array<T, Allocator<T>> buffer_;
};

template <typename VID_T, typename EDATA_T>
inline void sort_neighbors(Nbr<VID_T, EDATA_T>* begin,
                           Nbr<VID_T, EDATA_T>* end) {
  std::sort(begin, end,
            [](const Nbr<VID_T, EDATA_T>& lhs, const Nbr<VID_T, EDATA_T>& rhs) {
              return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
            });
}

template <typename VID_T, typename EDATA_T>
inline void sort_neighbors_tail(Nbr<VID_T, EDATA_T>* begin,
                                Nbr<VID_T, EDATA_T>* end, int unsorted,
                                std::vector<Nbr<VID_T, EDATA_T>>& buffer) {
  buffer.resize(unsorted);
  Nbr<VID_T, EDATA_T>* last = end - unsorted;
  std::move(last, end, buffer.data());
  std::sort(buffer.begin(), buffer.end(),
            [](const Nbr<VID_T, EDATA_T>& lhs, const Nbr<VID_T, EDATA_T>& rhs) {
              return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
            });
  Nbr<VID_T, EDATA_T>* src = last - 1;
  for (int i = unsorted - 1; i >= 0; --i) {
    auto& cur = buffer[i];
    while (src >= begin && cur.neighbor.GetValue() < src->neighbor.GetValue()) {
      *(--end) = std::move(*(src--));
    }
    *(--end) = std::move(cur);
  }
}

template <typename VID_T, typename EDATA_T>
inline Nbr<VID_T, EDATA_T>* sort_neighbors_tail_dedup(
    Nbr<VID_T, EDATA_T>* begin, Nbr<VID_T, EDATA_T>* end, int unsorted,
    std::vector<Nbr<VID_T, EDATA_T>>& buffer) {
  buffer.resize(unsorted);
  Nbr<VID_T, EDATA_T>* last = end - unsorted;
  std::move(last, end, buffer.data());
  std::sort(buffer.begin(), buffer.end(),
            [](const Nbr<VID_T, EDATA_T>& lhs, const Nbr<VID_T, EDATA_T>& rhs) {
              return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
            });
  Nbr<VID_T, EDATA_T>* src = last - 1;
  for (int i = unsorted - 1; i >= 0; --i) {
    auto& cur = buffer[i];
    if (i > 0 && cur.neighbor.GetValue() == buffer[i - 1].neighbor.GetValue()) {
      continue;
    }
    while (src >= begin && cur.neighbor.GetValue() < src->neighbor.GetValue()) {
      *(--end) = std::move(*(src--));
    }
    if (src >= begin && cur.neighbor.GetValue() == src->neighbor.GetValue()) {
      --src;
    }
    *(--end) = std::move(cur);
  }
  if (end + 1 != src) {
    while (src >= begin) {
      *(--end) = std::move(*(src--));
    }
    return end;
  } else {
    return begin;
  }
}

template <typename VID_T, typename EDATA_T>
inline Nbr<VID_T, EDATA_T>* sorted_dedup(Nbr<VID_T, EDATA_T>* begin,
                                         Nbr<VID_T, EDATA_T>* end) {
  int num = end - begin;
  int count = 0;
  for (int i = 1; i < num; ++i) {
    if (begin[i].neighbor.GetValue() == begin[i - 1].neighbor.GetValue()) {
      ++count;
    } else {
      begin[i - count].neighbor.SetValue(begin[i].neighbor.GetValue());
      begin[i - count].data = std::move(begin[i].data);
    }
  }
  return end - count;
}

template <typename VID_T, typename EDATA_T>
inline bool binary_remove_with_tomb(Nbr<VID_T, EDATA_T>* begin,
                                    Nbr<VID_T, EDATA_T>* end, VID_T target) {
  static constexpr VID_T tomb = std::numeric_limits<VID_T>::max();
  Nbr<VID_T, EDATA_T>* origin_begin = begin;
  Nbr<VID_T, EDATA_T>* origin_end = end;
  bool ret = false;
  while (begin != end) {
    int l2 = (end - begin) >> 1;
    Nbr<VID_T, EDATA_T>* mid = begin + l2;
    while (mid->neighbor.GetValue() == tomb && mid != end) {
      ++mid;
    }
    if (mid != end) {
      if (mid->neighbor.GetValue() < target) {
        begin = mid + 1;
      } else if (mid->neighbor.GetValue() == target) {
        ret = true;
        begin = mid + 1;
        while (mid->neighbor.GetValue() == target) {
          mid->neighbor.SetValue(tomb);
          if (mid == origin_begin) {
            break;
          }
          --mid;
        }
        break;
      } else {
        end = begin + l2;
      }
    } else {
      end = begin + l2;
    }
  }
  while (begin != origin_end && begin->neighbor.GetValue() == target) {
    ret = true;
    begin->neighbor.SetValue(tomb);
    ++begin;
  }
  return ret;
}

template <typename VID_T, typename EDATA_T>
inline bool binary_remove_one_with_tomb(Nbr<VID_T, EDATA_T>* begin,
                                        Nbr<VID_T, EDATA_T>* end,
                                        VID_T target) {
  static constexpr VID_T tomb = std::numeric_limits<VID_T>::max();
  while (begin != end) {
    int l2 = (end - begin) >> 1;
    Nbr<VID_T, EDATA_T>* mid = begin + l2;
    while (mid->neighbor.GetValue() == tomb && mid != end) {
      ++mid;
    }
    if (mid != end) {
      if (mid->neighbor.GetValue() < target) {
        begin = mid + 1;
      } else if (mid->neighbor.GetValue() == target) {
        mid->neighbor.SetValue(tomb);
        return true;
      } else {
        end = begin + l2;
      }
    } else {
      end = begin + l2;
    }
  }
  if (begin->neighbor.GetValue() == target) {
    begin->neighbor.SetValue(tomb);
    return true;
  }
  return false;
}

template <typename VID_T, typename EDATA_T>
inline Nbr<VID_T, EDATA_T>* remove_tombs(Nbr<VID_T, EDATA_T>* begin,
                                         Nbr<VID_T, EDATA_T>* end) {
  static constexpr VID_T tomb = std::numeric_limits<VID_T>::max();
  while (begin->neighbor.GetValue() != tomb && begin != end) {
    ++begin;
  }
  Nbr<VID_T, EDATA_T>* ptr = begin;
  while (ptr != end) {
    if (ptr->neighbor.GetValue() != tomb) {
      begin->neighbor.SetValue(ptr->neighbor.GetValue());
      begin->data = std::move(ptr->data);
      ++begin;
    }
    ++ptr;
  }
  return begin;
}

template <typename VID_T, typename EDATA_T>
inline void binary_update(Nbr<VID_T, EDATA_T>* begin, Nbr<VID_T, EDATA_T>* end,
                          VID_T target, const EDATA_T& value) {
  Nbr<VID_T, EDATA_T>* origin_begin = begin;
  Nbr<VID_T, EDATA_T>* origin_end = end;
  while (begin != end) {
    int l2 = (end - begin) >> 1;
    Nbr<VID_T, EDATA_T>* mid = begin + l2;
    if (mid->neighbor.GetValue() < target) {
      begin = mid + 1;
    } else if (mid->neighbor.GetValue() == target) {
      begin = mid + 1;
      while (mid->neighbor.GetValue() == target) {
        mid->data = value;
        if (mid == origin_begin) {
          break;
        }
        --mid;
      }
      break;
    } else {
      end = begin + l2;
    }
  }
  while (begin != origin_end && begin->neighbor.GetValue() == target) {
    begin->data = value;
  }
}

template <typename VID_T, typename EDATA_T>
inline void binary_update_one(Nbr<VID_T, EDATA_T>* begin,
                              Nbr<VID_T, EDATA_T>* end, VID_T target,
                              const EDATA_T& value) {
  while (begin != end) {
    int l2 = (end - begin) >> 1;
    Nbr<VID_T, EDATA_T>* mid = begin + l2;
    if (mid->neighbor.GetValue() < target) {
      begin = mid + 1;
    } else if (mid->neighbor.GetValue() == target) {
      mid->data = value;
      return;
    } else {
      end = begin + l2;
    }
  }
  if (begin->neighbor.GetValue() == target) {
    begin->data = value;
  }
}

template <typename VID_T, typename EDATA_T>
inline const Nbr<VID_T, EDATA_T>* binary_search_one(
    const Nbr<VID_T, EDATA_T>* begin, const Nbr<VID_T, EDATA_T>* end,
    VID_T target) {
  const Nbr<VID_T, EDATA_T>* original_end = end;
  while (begin != end) {
    int l2 = (end - begin) >> 1;
    const Nbr<VID_T, EDATA_T>* mid = begin + l2;
    if (mid->neighbor.GetValue() < target) {
      begin = mid + 1;
    } else if (mid->neighbor.GetValue() == target) {
      return mid;
    } else {
      end = begin + l2;
    }
  }
  if (begin != original_end && begin->neighbor.GetValue() == target) {
    return begin;
  }
  return original_end;
}

}  // namespace mutable_csr_impl

}  // namespace grape

#endif  // GRAPE_GRAPH_MUTABLE_CSR_IMPL_H_
