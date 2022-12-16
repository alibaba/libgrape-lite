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

// TODO: implement edgelist.cc

#ifndef GRIN_TOPOLOGY_EDGE_LIST_H_
#define GRIN_TOPOLOGY_EDGE_LIST_H_

#include "grin/predefine.h"

#ifdef ENABLE_EDGE_LIST

EdgeList get_edge_list(const Graph);

size_t get_edge_list_size(const EdgeList);

EdgeListIterator get_edge_list_begin(EdgeList);

EdgeListIterator get_next_edge_iter(EdgeList, const EdgeListIterator);

bool has_next_edge_iter(EdgeList, const EdgeListIterator);

Edge get_edge_from_iter(EdgeList, const EdgeListIterator);

EdgeListIterator get_first_adjacent_edge_iter(const Graph, EdgeList,
                                              const Vertex, const Direction);

bool get_next_adjacent_edge_iter(const Graph, EdgeList, EdgeListIterator*,
                                 const Direction);

EdgeList create_edge_list();

void destroy_edge_list(EdgeList);

bool insert_edge_to_list(EdgeList, const Edge);

#endif

#endif  // GRIN_TOPOLOGY_EDGE_LIST_H_
