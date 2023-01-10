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
#include "grin/include/topology/adjacentlist.h"

#ifdef ENABLE_ADJACENT_LIST

AdjacentList get_adjacent_list(const Graph gh, const Direction d, const Vertex v) {
    Graph_T* g = static_cast<Graph_T*>(gh);

    if (d == Direction::OUT) {
        AdjacentList al = new AdjacentList_T(g->GetOutgoingAdjList(Vertex_G(v)));
        return al;
    } else if (d == Direction::IN) {
        AdjacentList al = new AdjacentList_T(g->GetIncomingAdjList(Vertex_G(v)));
        return al;
    } else {
        return NULL_LIST;
    }
}

AdjacentListIterator get_adjacent_list_begin(const AdjacentList alh) {
    AdjacentList_T* al = static_cast<AdjacentList_T*>(alh);
    return al->begin_pointer();
}

AdjacentListIterator get_next_adjacent_iter(const AdjacentList alh, AdjacentListIterator alih) {
    AdjacentListIterator_T* ali = static_cast<AdjacentListIterator_T*>(alih);
    return ali++;
}

size_t get_adjacent_list_size(const AdjacentList alh) {
    AdjacentList_T* al = static_cast<AdjacentList_T*>(alh);
    return al->Size();
}

bool has_next_adjacent_iter(const AdjacentList alh, const AdjacentListIterator alih) {
    AdjacentList_T* al = static_cast<AdjacentList_T*>(alh);
    AdjacentListIterator_T* ali = static_cast<AdjacentListIterator_T*>(alih);
    return ali != al->end_pointer();
}

Vertex get_neighbor_from_iter(const AdjacentList alh, const AdjacentListIterator alih) {
    AdjacentListIterator_T* ali = static_cast<AdjacentListIterator_T*>(alih);
    return ali->get_neighbor_lid();
}

#ifdef WITH_EDGE_WEIGHT
DataType get_adjacent_edge_weight_type(const Graph g) {
    return DataTypeName<G_EDATA_T>::Get();
}

G_EDATA_T get_adjacent_edge_weight_value(const AdjacentList alh, const AdjacentListIterator alih) {
    AdjacentListIterator_T* ali = static_cast<AdjacentListIterator_T*>(alih);
    return ali->get_data();
}
#endif

#endif