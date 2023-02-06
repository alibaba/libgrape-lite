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
#include "grin/include/topology/vertexlist.h"
}

#ifdef ENABLE_VERTEX_LIST

VertexList get_vertex_list(const Graph gh) {
  Graph_T* g = static_cast<Graph_T*>(gh);
  VertexList vl = new VertexList_T(g->Vertices());
  return vl;
}

void destroy_vertex_list(VertexList vlh) {
  if (vlh != NULL_LIST) {
    VertexList_T* vl = static_cast<VertexList_T*>(vlh);
    delete vl;
  }
}

size_t get_vertex_list_size(const VertexList vlh) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  return vl->size();
}

VertexListIterator get_vertex_list_begin(const VertexList vlh) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  VertexListIterator_T* vli = new VertexListIterator_T(vl->begin_value());
  return vli;
}

void destroy_vertex_list_iterator(VertexListIterator vlih) {
  VertexListIterator_T* vli = static_cast<VertexListIterator_T*>(vlih);
  delete vli;
}

VertexListIterator get_next_vertex_iter(const VertexList vlh,
                                        VertexListIterator vlih) {
  VertexListIterator_T* vli = static_cast<VertexListIterator_T*>(vlih);
  (*vli)++;
  return vli;
}

bool has_next_vertex_iter(const VertexList vlh, VertexListIterator vlih) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  VertexListIterator_T* vli = static_cast<VertexListIterator_T*>(vlih);
  return (*vli) < vl->end_value();
}

Vertex get_vertex_from_iter(const VertexList vlh, VertexListIterator vlih) {
  VertexListIterator_T* vli = static_cast<VertexListIterator_T*>(vlih);
  Vertex_T* v = new Vertex_T(*vli);
  return v;
}

#ifdef CONTINUOUS_VID_TRAIT
VertexID get_begin_vertex_id_from_list(const VertexList vlh) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  VertexID_T* vi = new VertexID_T(vl->begin_value());
  return vi;
}

VertexID get_end_vertex_id_from_list(const VertexList vlh) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  VertexID_T* vi = new VertexID_T(vl->end_value());
  return vi;
}
#endif

#ifdef MUTABLE_GRAPH
VertexList create_vertex_list() { return NULL_LIST }
bool insert_vertex_to_list(VertexList vlh, const Vertex vh) { return false }
#endif
#endif
