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

#ifndef GRAPE_UTILS_ITERATOR_PAIR_H_
#define GRAPE_UTILS_ITERATOR_PAIR_H_

namespace grape {

/**
 * @brief IteratorPair is a wrapper for begin and end iterators.
 *
 * @tparam T The type of data type will be iterated over.
 */
template <typename T>
class IteratorPair {
 public:
  IteratorPair(const T begin, const T end) : begin_(begin), end_(end) {}

  inline T begin() const { return begin_; }
  inline T end() const { return end_; }

  IteratorPair& operator=(const IteratorPair& ip) {
    this->begin_ = ip.begin();
    this->end_ = ip.end();
    return *this;
  }

  bool empty() const { return begin_ == end_; }

  void set_begin(T begin) { begin_ = begin; }

  void set_end(T end) { end_ = end; }

  int size() const { return end_ - begin_; }

 private:
  T begin_, end_;
};

}  // namespace grape

#endif  // GRAPE_UTILS_ITERATOR_PAIR_H_
