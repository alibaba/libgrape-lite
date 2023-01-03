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

#include "grin/src/predefine.h"

#ifdef ENABLE_EDGE_LIST

EdgeList get_edge_list(const Graph g, const Direction d) {
    EdgeListHandler eh;
    eh.g = g;
    eh.size = 0;
    eh.d = d;
    eh.vlist = g->InnerVertices();

    if (d == Direction::IN || d == Direction::BOTH) {
        grape::AdjList<VID_T, EDATA_T> in_adj_list = grape::AdjList<VID_T, EDATA_T>(g->GetIncomingAdjList(*eh.vlist.begin()).begin_pointer(), g->GetIncomingAdjList(*eh.vlist.end()).end_pointer());
        eh.size += in_adj_list.Size();
    }
    if (d == Direction::OUT || d == Direction::BOTH) {
        grape::AdjList<VID_T, EDATA_T> out_adj_list = grape::AdjList<VID_T, EDATA_T>(g->GetOutgoingAdjList(*eh.vlist.begin()).begin_pointer(), g->GetOutgoingAdjList(*eh.vlist.end()).end_pointer());
        eh.size += out_adj_list.Size();
    }
    return &eh;
}

size_t get_edge_list_size(const EdgeList eh) {
    return eh->size;
}

void adjust_edgelist_iter(const EdgeList eh, EdgeListIterator eiter) {
    bool flag = true;
    while (eiter->viter != eh->vlist.end()) {
        flag = true;
        if (eh->g->GetOutgoingAdjList(*(eiter->viter)).end_pointer() == eiter->ptr) {
            assert(eh->d != Direction::IN);
            flag = false;
            if (eh->d == Direction::BOTH) {
                eiter->ptr = eh->g->GetIncomingAdjList(*eiter->viter).begin_pointer();
                eiter->d = Direction::IN;
            } else {
                eiter->viter++;
                eiter->ptr = eh->g->GetOutgoingAdjList(*eiter->viter).begin_pointer();
            }
        }
        if (eh->g->GetIncomingAdjList(*(eiter->viter)).end_pointer() == eiter->ptr) {
            assert(eh->d != Direction::OUT);
            flag = false;
            eiter->viter++;
            if (eh->d == Direction::BOTH) {
                eiter->ptr = eh->g->GetOutgoingAdjList(*eiter->viter).begin_pointer();
                eiter->d = Direction::OUT;
            } else {
                eiter->ptr = eh->g->GetIncomingAdjList(*eiter->viter).begin_pointer();
            }
        }
        if (flag) break;
    }
}

EdgeListIterator get_edge_list_begin(const EdgeList eh) {
    EdgeListIteratorHandler eih;
    eih.viter = eh->vlist.begin();
    if (eh->d == Direction::IN) {
        eih.ptr = eh->g->GetIncomingAdjList(*eih.viter).begin_pointer();
        eih.d = Direction::IN;
    } else {
        eih.ptr = eh->g->GetOutgoingAdjList(*eih.viter).begin_pointer();
        eih.d = Direction::OUT;
    }
    adjust_edgelist_iter(eh, &eih);
    return &eih;
}

EdgeListIterator get_next_edge_iter(const EdgeList eh, const EdgeListIterator eiter) {
    EdgeListIterator niter = eiter;
    niter->ptr++;
    adjust_edgelist_iter(eh, niter);
    return niter;
}

bool has_next_edge_iter(const EdgeList eh, const EdgeListIterator eiter) {
    return eiter->viter != eh->vlist.end();
}

Edge get_edge_from_iter(const EdgeList eh, const EdgeListIterator eiter) {
    if (eiter->d == Direction::IN) {
        return Edge(eiter->ptr->get_neighbor().GetValue(), (*eiter->viter).GetValue(), eiter->ptr->get_data());
    } else {
        return Edge((*eiter->viter).GetValue(), eiter->ptr->get_neighbor().GetValue(), eiter->ptr->get_data());
    }
}

EdgeList get_adjacent_list(const Graph g, const Direction d, const Vertex v) {
    EdgeListHandler eh;
    eh.g = g;
    eh.size = 0;
    eh.d = d;
    eh.vlist = VertexList(v.GetValue(), v.GetValue());

    if (d == Direction::IN || d == Direction::BOTH) {
        eh.size += g->GetIncomingAdjList(v).Size();
    }
    if (d == Direction::OUT || d == Direction::BOTH) {
        eh.size += g->GetOutgoingAdjList(v).Size();
    }
    return &eh;
}

//EdgeList create_edge_list();

//void destroy_edge_list(EdgeList);

//bool insert_edge_to_list(EdgeList, const Edge);

#endif

#endif  // GRIN_TOPOLOGY_EDGE_LIST_H_
