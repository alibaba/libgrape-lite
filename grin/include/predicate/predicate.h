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

// This header file is not available for libgrape-lite.

#ifndef GRIN_PREDICATE_PREDICATE_H_
#define GRIN_PREDICATE_PREDICATE_H_

#include "../predefine.h"

#ifdef ENABLE_PREDICATE

VertexListIterator get_first_vertex_iter_with_predicate(VertexList,
                                                        const void*);

VertexListIterator get_next_vertex_iter_with_predicate(VertexList,
                                                       const VertexListIterator,
                                                       const void*);

EdgeListIterator get_first_edge_iter_with_predicate(EdgeList, const void*);

EdgeListIterator get_next_edge_iter_with_predicate(EdgeList,
                                                   const EdgeListIterator,
                                                   const void*);

EdgeListIterator get_first_adjacent_edge_iter_with_predicate(
    const Graph, EdgeList, const Vertex, const Direction, const void*);

bool get_next_adjacent_edge_iter_with_predicate(const Graph, EdgeList,
                                                EdgeListIterator*,
                                                const Direction, const void*);

bool verify_vertex(const Graph, const Vertex, const void*);

bool verify_edge(const Graph, const Edge, const void*);

VertexVerifyResults batch_verify_vertices(const Graph, const VertexList,
                                          const void*);

bool get_vertex_verify_result(const Graph, const VertexVerifyResults,
                              const Vertex);

EdgeVerifyResults batch_verify_edges(const Graph, const EdgeList, const void*);

bool get_edge_verify_result(const Graph, const EdgeVerifyResults, const Edge);

#endif

#endif  // GRIN_PREDICATE_PREDICATE_H_
