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

#ifndef GRAPE_FRAGMENT_FRAGMENT_BASE_H_
#define GRAPE_FRAGMENT_FRAGMENT_BASE_H_

#include <vector>

#include "grape/fragment/id_parser.h"
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/vertex_map/vertex_map.h"
#include "grape/worker/comm_spec.h"

namespace grape {

struct PrepareConf {
  MessageStrategy message_strategy;
  bool need_split_edges;
  bool need_split_edges_by_fragment;
  bool need_mirror_info;
  bool need_build_device_vm;
};

/**
 * @brief FragmentBase is the base class for fragments.
 *
 * @note: The pure virtual functions in the class work as interfaces,
 * instructing sub-classes to implement. The override functions in the
 * derived classes would be invoked directly, not via virtual functions.
 *
 * @tparam OID_T
 * @tparam VDATA_T
 * @tparam EDATA_T
 */
template <typename OID_T, typename VDATA_T, typename EDATA_T>
class FragmentBase {
 public:
  FragmentBase() {}
  virtual ~FragmentBase() {}

 protected:
  void init(fid_t fid, fid_t fnum, bool directed) {
    fid_ = fid;
    fnum_ = fnum;
    directed_ = directed;
  }

 public:
  /**
   * @brief Returns true if the fragment is directed, false otherwise.
   *
   * @return true if the fragment is directed, false otherwise.
   */
  bool directed() const { return directed_; }

  /**
   * @brief Returns the ID of this fragment.
   *
   * @return The ID of this fragment.
   */
  fid_t fid() const { return fid_; }

  /**
   * @brief Returns the number of fragments.
   *
   * @return The number of fragments.
   */
  fid_t fnum() const { return fnum_; }

  /**
   * @brief Returns the number of edges in this fragment.
   *
   * @return The number of edges in this fragment.
   */
  virtual size_t GetEdgeNum() const = 0;

  /**
   * @brief Returns the number of vertices in this fragment.
   *
   * @return The number of vertices in this fragment.
   */
  virtual size_t GetVerticesNum() const = 0;

  /**
   * @brief Returns the number of vertices in the entire graph.
   *
   * @return The number of vertices in the entire graph.
   */
  virtual size_t GetTotalVerticesNum() const = 0;

  /**
   * @brief For some kind of applications, specific data structures will be
   * generated.
   *
   * @param strategy
   * @param need_split_edge
   */
  virtual void PrepareToRunApp(const CommSpec& comm_spec, PrepareConf conf) = 0;

 protected:
  template <typename IOADAPTOR_T>
  void serialize(std::unique_ptr<IOADAPTOR_T>& writer) {
    InArchive arc;
    arc << fid_ << fnum_ << directed_;
    CHECK(writer->WriteArchive(arc));
  }

  template <typename IOADAPTOR_T>
  void deserialize(std::unique_ptr<IOADAPTOR_T>& reader) {
    OutArchive arc;
    CHECK(reader->ReadArchive(arc));
    arc >> fid_ >> fnum_ >> directed_;
  }

  fid_t fid_, fnum_;
  bool directed_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_FRAGMENT_BASE_H_
