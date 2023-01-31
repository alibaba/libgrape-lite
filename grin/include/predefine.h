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
#include <climits>

#include "grape/fragment/immutable_edgecut_fragment.h"

#ifndef GRIN_INCLUDE_PREDEFINE_H_
#define GRIN_INCLUDE_PREDEFINE_H_

// The enum type for edge directions.
typedef enum {
  IN = 0,
  OUT = 1,
  BOTH = 2,
} Direction;

// The enum type for partition strategies.
typedef enum {
  VERTEX_CUT = 0,
  EDGE_CUT = 1,
  HYBRID = 2,
} PartitionStrategy;

// The enum type for replicate-edge strategies.
typedef enum {
  ALL = 0,
  PART = 1,
  NONE = 2,
} ReplicateEdgeStrategy;

// The enum type for vertex/edge data type.
typedef enum {
  INT = 0,
  UNSIGNED = 1,
  FLOAT = 2,
  DOUBLE = 3,
  EMPTY = 4,
  OTHER = 5,
} DataType;

template <typename T>
struct DataTypeName {
  static DataType Get() { return DataType::OTHER; }
};
template <>
struct DataTypeName<int> {
  static DataType Get() { return DataType::INT; }
};
template <>
struct DataTypeName<unsigned> {
  static DataType Get() { return DataType::UNSIGNED; }
};
template <>
struct DataTypeName<float> {
  static DataType Get() { return DataType::FLOAT; }
};
template <>
struct DataTypeName<double> {
  static DataType Get() { return DataType::DOUBLE; }
};
template <>
struct DataTypeName<grape::EmptyType> {
  static DataType Get() { return DataType::EMPTY; }
};

/* The following macros are defined as the features of the storage. */
#define WITH_VERTEX_DATA             // There is data on vertex.
#define WITH_EDGE_DATA               // There is data on edge, e.g. weight.
#define ENABLE_VERTEX_LIST           // Enable the vertex list structure.
#define ENABLE_ADJACENT_LIST         // Enable the adjacent list structure.
// Note: edge_list is only used in vertex_cut fragment
// #define ENABLE_EDGE_LIST          // Enable the edge list structure.

// The partition strategy.
#define PARTITION_STRATEGY EDGE_CUT
// There are all/part edges on local vertices.
#define EDGES_ON_LOCAL_VERTEX ALL
// There are all/part/none edges on non-local vertices.
#define EDGES_ON_NON_LOCAL_VERTEX NONE
// The direction of edges on local vertices.
#define EDGES_ON_LOCAL_VERTEX_DIRECTION BOTH

/* The followings macros are defined as invalid value. */
#define NULL_TYPE NULL        // Null type (null data type)
#define NULL_GRAPH NULL       // Null graph handle (invalid return value).
#define NULL_VERTEX NULL      // Null vertex handle (invalid return value).
#define NULL_EDGE NULL        // Null edge handle (invalid return value).
#define NULL_PARTITION NULL   // Null partition handle (invalid return value).
#define NULL_LIST NULL        // Null list of any kind.
#define NULL_REMOTE_PARTITION NULL  // same as partition.
#define NULL_REMOTE_VERTEX NULL     // same as vertex.
#define NULL_REMOTE_EDGE NULL       // same as edge.

/* The following data types shall be defined through typedef. */
// local graph
typedef grape::ImmutableEdgecutFragment<G_OID_T, G_VID_T, G_VDATA_T, G_EDATA_T>
    Graph_T;
typedef void* Graph;

// vertex
typedef Graph_T::vid_t Vertex;
typedef Graph_T::vertex_t Vertex_G;

// vertex list
typedef Graph_T::vertex_range_t VertexList_T;
typedef void* VertexList;
typedef Graph_T::vid_t VertexListIterator;

// vertex data
#ifdef WITH_VERTEX_DATA
typedef Graph_T::vdata_t VertexData;
#endif

// adjacent list
typedef Graph_T::adj_list_t AdjacentList_T;
typedef void* AdjacentList;
typedef Graph_T::nbr_t AdjacentListIterator_T;
typedef void* AdjacentListIterator;

// edge
typedef Graph_T::edge_t Edge_T;
typedef void* Edge;

// edge list
struct EdgeList_T {
  Graph_T* g;
  Direction d;
  size_t size;
  VertexList_T* vl;
};
typedef void* EdgeList;
struct EdgeListIterator_T {
  Direction d;
  VertexListIterator vli;
  AdjacentListIterator_T* ptr;
};
typedef void* EdgeListIterator;

#ifdef WITH_EDGE_DATA
typedef Graph_T::edata_t EdgeData;
#endif

// partitioned graph
typedef Graph_T PartitionedGraph_T;
typedef void* PartitionedGraph;

// partition and partition list
typedef unsigned Partition;
typedef Partition* PartitionList;

// remote partition and remote partition list
typedef Partition RemotePartition;
typedef PartitionList RemotePartitionList;

// remote vertex and remote vertex list
typedef Vertex RemoteVertex;
typedef RemoteVertex* RemoteVertexList;

// remote edge
typedef Edge_T RemoteEdge_T;
typedef Edge RemoteEdge;

#endif  // GRIN_INCLUDE_PREDEFINE_H_
