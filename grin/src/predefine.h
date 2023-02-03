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
#include "grin/include/predefine.h"
#include "grape/fragment/immutable_edgecut_fragment.h"

#ifndef GRIN_SRC_PREDEFINE_H_
#define GRIN_SRC_PREDEFINE_H_

template <typename T>
struct DataTypeName {
  static DataType Get() { return DataType::OTHER; }
};
template <>
struct DataTypeName<int32_t> {
  static DataType Get() { return DataType::INT; }
};
template <>
struct DataTypeName<int64_t> {
  static DataType Get() { return DataType::LONG; }
};
template <>
struct DataTypeName<uint32_t> {
  static DataType Get() { return DataType::UNSIGNED; }
};
template <>
struct DataTypeName<uint64_t> {
  static DataType Get() { return DataType::UNSIGNED_LONG; }
};
template <>
struct DataTypeName<float> {
  static DataType Get() { return DataType::FLOAT; }
};
template <>
struct DataTypeName<double> {
  static DataType Get() { return DataType::DOUBLE; }
};


/* The following data types shall be defined through typedef. */
// local graph
typedef grape::ImmutableEdgecutFragment<G_OID_T, G_VID_T, G_VDATA_T, G_EDATA_T>
    Graph_T;

// vertex
typedef Graph_T::vertex_t Vertex_T;
typedef Graph_T::vid_t VertexID_T;
typedef Graph_T::vdata_t VertexData_T;

// vertex list
#ifdef ENABLE_VERTEX_LIST
typedef Graph_T::vertex_range_t VertexList_T;
typedef Graph_T::vid_t VertexListIterator_T;
#endif

// indexed vertex list
#ifdef ENABLE_INDEXED_VERTEX_LIST
typedef Graph_T::vid_t VertexIndex_T;
#endif

// adjacent list
#ifdef ENABLE_ADJACENT_LIST
typedef Graph_T::adj_list_t AdjacentList_T;
typedef Graph_T::nbr_t AdjacentListIterator_T;
#endif

// edge
typedef Graph_T::edge_t Edge_T;
typedef Graph_T::edata_t EdgeData_T;

// partitioned graph
typedef Graph_T PartitionedGraph_T;

// remote vertex
typedef Vertex_T RemoteVertex_T;

// remote edge
typedef Edge_T RemoteEdge_T;

#endif  // GRIN_SRC_PREDEFINE_H_
