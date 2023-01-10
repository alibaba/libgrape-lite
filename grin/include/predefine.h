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
#include "grape/graph/adj_list.h"
#include "grape/graph/edge.h"
#include "grape/graph/vertex.h"
#include "grape/utils/vertex_array.h"

#ifndef GRIN_PRE_DEFINE_H_
#define GRIN_PRE_DEFINE_H_

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

typedef enum {
  INT = 0,
  UNSIGNED = 1,
  FLOAT = 2,
  DOUBLE = 3,
  EMPTY = 4,
  OTHER = 5,
} DataType;

template <typename T>
struct DataTypeName
{
    static DataType Get()
    {
        return DataType::OTHER;
    }
};
template <>
struct DataTypeName<int>
{
    static DataType Get()
    {
        return DataType::INT;
    }
};
template <>
struct DataTypeName<unsigned>
{
    static DataType Get()
    {
        return DataType::UNSIGNED;
    }
};
template <>
struct DataTypeName<float>
{
    static DataType Get()
    {
        return DataType::FLOAT;
    }
};
template <>
struct DataTypeName<double>
{
    static DataType Get()
    {
        return DataType::DOUBLE;
    }
};
template <>
struct DataTypeName<grape::EmptyType>
{
    static DataType Get()
    {
        return DataType::EMPTY;
    }
};


/* The following macros are defined as the features of the storage. */
#define WITH_VERTEX_DATA               // There is data on vertex.
#define WITH_EDGE_SRC                  // There is src data for edge.
#define WITH_EDGE_DST                  // There is dst data for edge.
#define WITH_EDGE_WEIGHT               // There is weight for edge.
#define ENABLE_VERTEX_LIST             // Enable the vertex list structure.
#define ENABLE_EDGE_LIST               // Enable the edge list structure.
#define ENABLE_ADJACENT_LIST           // Enable the adjacent list structure.
#define PARTITION_STRATEGY EDGE_CUT  // The partition strategy.
// There are all/part edges on local vertices.
#define EDGES_ON_LOCAL_VERTEX ALL
// There are all/part/none edges on non-local vertices.
#define EDGES_ON_NON_LOCAL_VERTEX NONE
// The direction of edges on local vertices.
#define EDGES_ON_LOCAL_VERTEX_DIRECTION BOTH


/* The followings macros are defined as invalid value. */
#define NULL_TYPE NULL                  // Null type (null data type).
#define NULL_GRAPH NULL                 // Null graph handle (invalid return value).
#define NULL_VERTEX UINT_MAX            // Null vertex handle (invalid return value).
#define NULL_EDGE NULL                  // Null edge handle (invalid return value).
#define NULL_PARTITION UINT_MAX         // Null partition handle (invalid return value).
#define NULL_REMOTE_PARTITION UINT_MAX  // Null remote partition handle (invalid return value).
#define NULL_REMOTE_VERTEX UINT_MAX     // Null remote vertex handle (invalid return value).
#define NULL_REMOTE_EDGE NULL           // Null remote edge handle (invalid return value).
#define NULL_LIST NULL                  // Null list of any kind handler

/* The following data types shall be defined through typedef. */

// local graph
typedef grape::ImmutableEdgecutFragment<G_OID_T, G_VID_T, G_VDATA_T, G_EDATA_T> Graph_T;
typedef void* Graph;

// vertex
typedef G_VID_T Vertex;
typedef grape::Vertex<G_VID_T> Vertex_G;

// vertex list
typedef grape::VertexRange<G_VID_T> VertexList_T;  
typedef void* VertexList;
typedef G_VID_T VertexListIterator;

// adjacent list
typedef grape::AdjList<G_VID_T, G_EDATA_T> AdjacentList_T;
typedef void* AdjacentList;
typedef grape::Nbr<G_VID_T, G_EDATA_T> AdjacentListIterator_T;
typedef void* AdjacentListIterator;

// edge
typedef grape::Edge<G_VID_T, G_EDATA_T> Edge_T;        
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

// partitioned graph
typedef grape::ImmutableEdgecutFragment<G_OID_T, G_VID_T, G_VDATA_T, G_EDATA_T> PartitionedGraph_T;
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


/* The followings macros are not required in libgrape-lite. */
/*
#define WITH_VERTEX_LABEL          // There are labels on vertices.
#define WITH_VERTEX_PROPERTY       // There is any property on vertices.
#define WITH_VERTEX_PRIMARTY_KEYS  // There are primary keys for vertex.
#define WITH_EDGE_LABEL            // There are labels for edges.
#define WITH_EDGE_PROPERTY         // There is any property for edges.
#define COLUMN_STORE               // Column-oriented storage for properties.
#define PROPERTY_ON_NON_LOCAL_VERTEX
// There are properties on non-local vertices.
#define PROPERTY_ON_NON_LOCAL_EDGE  // There are properties on non-local edges.
#define EDGES_ON_NON_LOCAL_VERTEX_DIRECTION
// The direction of edges on on-local vertices.
#define ENABLE_PREDICATE  // Enable predicates for vertices and edges.

#define NULL_VERTEX_LABEL  // Null vertex label handle (invalid return value).
#define NULL_EDGE_LABEL    // Null vertex label handle (invalid return value).
#define NULL_PROPERTY      // Null property handle (invalid return value).
#define NULL_ROW           // Null row handle (invalid return value).
*/

/* The followings data types are not required in libgrape-lite. */
/*
typedef void* VertexLabel;  // This is a handle for accessing a vertex label.
typedef void* VertexLabelList;  // The type for a list of vertex labels.
typedef void* EdgeLabel;        // This is a handle for accessing a edge label.
typedef void* EdgeLabelList;    // The type for a list of edge labels.
typedef void* Property;  // This is a handle for accessing a single property.
typedef void* PropertyList;  // The type for respresenting a list of properties.
typedef void* Row;           // This is a handle for a row (key-value paris).
typedef void* RowList;       // The type for respresenting a list of rows.
typedef void* RowListIterator;  // The iterator for accessing items in RowList.
typedef void* VertexVerifyResults;  // The type for vertices' verify results.
typedef void* EdgeVerifyResults;    // The type for edges' verify results.
*/

#endif  // GRIN_PRE_DEFINE_H_