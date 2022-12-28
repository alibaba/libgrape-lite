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

// TODO: implement vertexlist.cc

#ifndef GRIN_TOPOLOGY_VERTEX_LIST_H_
#define GRIN_TOPOLOGY_VERTEX_LIST_H_

#include "../predefine.h"

#ifdef ENABLE_VERTEX_LIST

VertexList get_vertex_list(const Graph);

size_t get_vertex_list_size(const VertexList);

VertexListIterator get_vertex_list_begin(VertexList);

VertexListIterator get_next_vertex_iter(VertexList, const VertexListIterator);

bool has_next_vertex_iter(VertexList, const VertexListIterator);

Vertex get_vertex_from_iter(VertexList, const VertexListIterator);

VertexList create_vertex_list();

void destroy_vertex_list(VertexList);

bool insert_vertex_to_list(VertexList, const Vertex);

#endif

#endif  // GRIN_TOPOLOGY_VERTEX_LIST_H_
