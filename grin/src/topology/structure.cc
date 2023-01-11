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
#include "grin/include/topology/structure.h"
}

bool is_directed(const Graph gh) {
    Graph_T* g = static_cast<Graph_T*>(gh);
    return g->directed();
}

#ifdef WITH_EDGE_SRC
Vertex get_edge_src(const Graph gh, const Edge eh) {
    Edge_T* e = static_cast<Edge_T*>(eh);
    return e->src;
}
#endif

#ifdef WITH_EDGE_DST
Vertex get_edge_dst(const Graph gh, const Edge eh) {
    Edge_T* e = static_cast<Edge_T*>(eh);
    return e->dst;
}
#endif

#ifdef WITH_EDGE_WEIGHT
DataType get_edge_weight_type(const Graph g) {
    return DataTypeName<G_EDATA_T>::Get();
}

G_EDATA_T get_edge_weight_value(const Graph gh, const Edge eh) {
    Edge_T* e = static_cast<Edge_T*>(eh);
    return e->edata;
}
#endif

#ifdef WITH_VERTEX_DATA
DataType get_vertex_data_type(const Graph g) {
    return DataTypeName<G_VDATA_T>::Get(); 
}

G_VDATA_T get_vertex_data_value(const Graph gh, const Vertex v) {
    Graph_T* g = static_cast<Graph_T*>(gh);
    return g->GetData(Vertex_G(v));
}

//void set_vertex_data_value(const Graph g, Vertex, const void*);
#endif