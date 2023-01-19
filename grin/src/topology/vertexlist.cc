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

extern "C" {
#include "grin/include/topology/vertexlist.h"
}

#ifdef ENABLE_VERTEX_LIST

VertexList get_vertex_list(const Graph gh) {
  Graph_T* g = static_cast<Graph_T*>(gh);
  VertexList vl = new VertexList_T(g->Vertices());
  return vl;
}

size_t get_vertex_list_size(const VertexList vlh) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  return vl->size();
}

VertexListIterator get_vertex_list_begin(const VertexList vlh) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  return vl->begin_value();
}

VertexListIterator get_next_vertex_iter(const VertexList vlh,
                                        VertexListIterator vlih) {
  return vlih++;
}

bool has_next_vertex_iter(const VertexList vlh, VertexListIterator vlih) {
  VertexList_T* vl = static_cast<VertexList_T*>(vlh);
  return vlih < vl->end_value();
}

Vertex get_vertex_from_iter(const VertexList vlh, VertexListIterator vlih) {
  return vlih;
}

#endif
