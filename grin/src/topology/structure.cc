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

#include "grin/src/predefine.h"

extern "C" {
#include "grin/include/topology/structure.h"
}

bool is_directed(const Graph gh) {
  Graph_T* g = static_cast<Graph_T*>(gh);
  return g->directed();
}

size_t get_vertex_num(const Graph gh) {
  Graph_T* g = static_cast<Graph_T*>(gh);
  return g->GetVerticesNum();
}

size_t get_edge_num(const Graph gh) {
  Graph_T* g = static_cast<Graph_T*>(gh);
  return g->GetEdgeNum();
}

void destroy_vertex(Vertex vh) {
  Vertex_T* v = static_cast<Vertex_T*>(vh);
  delete v;
}

DataType get_vertex_id_data_type(const Graph gh) {
  return DataTypeName<VertexID_T>::Get();
}

VertexID get_vertex_id(const Graph gh, const Vertex vh) {
  Vertex_T* v = static_cast<Vertex_T*>(vh);
  VertexID_T* id = new VertexID_T(v->GetValue());
  return id;
}

#ifdef WITH_VERTEX_DATA
DataType get_vertex_data_type(const Graph g) {
  return DataTypeName<VertexData>::Get();
}

VertexData get_vertex_data_value(const Graph gh, const Vertex vh) {
  Graph_T* g = static_cast<Graph_T*>(gh);
  Vertex_T* v = static_cast<Vertex_T*>(vh);
  VertexData_T* vdata = new VertexData_T(g->GetData(*v));
  return vdata;
}

void destroy_vertex_data(VertexData vdh) {
  VertexData_T* vdata = static_cast<VertexData_T*>(vdh);
  delete vdata;
}

#ifdef MUTABLE_GRAPH
void set_vertex_data_value(Graph gh, Vertex vh, const VertexData vdata){
  return;
}
#endif
#endif

void destroy_edge(Edge eh) {
  Edge_T* e = static_cast<Edge_T*>(eh);
  delete e;
}

Vertex get_edge_src(const Graph gh, const Edge eh) {
  Edge_T* e = static_cast<Edge_T*>(eh);
  Vertex_T* v = new Vertex_T(e->src);
  return v;
}

Vertex get_edge_dst(const Graph gh, const Edge eh) {
  Edge_T* e = static_cast<Edge_T*>(eh);
  Vertex_T* v = new Vertex_T(e->dst);
  return v;
}

#ifdef WITH_EDGE_DATA
DataType get_edge_data_type(const Graph g) {
  return DataTypeName<EdgeData>::Get();
}

EdgeData get_edge_data_value(const Graph gh, const Edge eh) {
  Edge_T* e = static_cast<Edge_T*>(eh);
  EdgeData_T* edata = new EdgeData_T(e->edata);
  return edata;
}

void destroy_edge_data(EdgeData edh) {
  EdgeData_T* edata = static_cast<EdgeData_T*>(edh);
  delete edata;
}

#ifdef MUTABLE_GRAPH
void set_edge_data_value(Graph gh, Edge eh, const EdgeData edata){
  return;
}
#endif
#endif
