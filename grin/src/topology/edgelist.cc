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
#include "grin/include/topology/edgelist.h"
#include "grin/include/topology/vertexlist.h"
}

#ifdef ENABLE_EDGE_LIST

EdgeList get_edge_list(const Graph gh, const Direction d) {
  EdgeList_T* el = new EdgeList_T();
  el->g = static_cast<Graph_T*>(gh);
  el->size = 0;
  el->d = d;
  el->vl = new VertexList_T(el->g->InnerVertices());

  Vertex begin = el->vl->begin_value(), end = el->vl->end_value() - 1;
  if (d == Direction::IN || d == Direction::BOTH) {
    AdjacentList_T in_adj_list = AdjacentList_T(
        el->g->GetIncomingAdjList(Vertex_G(begin)).begin_pointer(),
        el->g->GetIncomingAdjList(Vertex_G(end)).end_pointer());
    el->size += in_adj_list.Size();
  }
  if (d == Direction::OUT || d == Direction::BOTH) {
    AdjacentList_T out_adj_list = AdjacentList_T(
        el->g->GetOutgoingAdjList(Vertex_G(begin)).begin_pointer(),
        el->g->GetOutgoingAdjList(Vertex_G(end)).end_pointer());
    el->size += out_adj_list.Size();
  }
  return el;
}

size_t get_edge_list_size(const EdgeList elh) {
  EdgeList_T* el = static_cast<EdgeList_T*>(elh);
  return el->size;
}

void adjust_edgelist_iter(const EdgeList elh, EdgeListIterator elih) {
  EdgeList_T* el = static_cast<EdgeList_T*>(elh);
  EdgeListIterator_T* eli = static_cast<EdgeListIterator_T*>(elih);
  bool flag = true;
  while (has_next_vertex_iter(el->vl, eli->vli)) {
    flag = true;
    Vertex_G gv = Vertex_G(get_vertex_from_iter(el->vl, eli->vli));
    if (el->g->GetOutgoingAdjList(gv).end_pointer() == eli->ptr) {
      assert(el->d != Direction::IN);
      flag = false;
      if (el->d == Direction::BOTH) {
        eli->ptr = el->g->GetIncomingAdjList(gv).begin_pointer();
        eli->d = Direction::IN;
      } else {
        eli->vli++;
        eli->ptr = el->g->GetOutgoingAdjList(gv).begin_pointer();
      }
    }
    if (el->g->GetIncomingAdjList(gv).end_pointer() == eli->ptr) {
      assert(el->d != Direction::OUT);
      flag = false;
      eli->vli++;
      if (el->d == Direction::BOTH) {
        eli->ptr = el->g->GetOutgoingAdjList(gv).begin_pointer();
        eli->d = Direction::OUT;
      } else {
        eli->ptr = el->g->GetIncomingAdjList(gv).begin_pointer();
      }
    }
    if (flag)
      break;
  }
}

EdgeListIterator get_edge_list_begin(const EdgeList elh) {
  EdgeList_T* el = static_cast<EdgeList_T*>(elh);
  EdgeListIterator_T* eli = new EdgeListIterator_T();

  eli->vli = el->vl->begin_value();
  Vertex_G gv = Vertex_G(get_vertex_from_iter(el->vl, eli->vli));

  if (el->d == Direction::IN) {
    eli->ptr = el->g->GetIncomingAdjList(gv).begin_pointer();
    eli->d = Direction::IN;
  } else {
    eli->ptr = el->g->GetOutgoingAdjList(gv).begin_pointer();
    eli->d = Direction::OUT;
  }
  adjust_edgelist_iter(el, eli);
  return eli;
}

EdgeListIterator get_next_edge_iter(const EdgeList elh, EdgeListIterator elih) {
  EdgeList_T* el = static_cast<EdgeList_T*>(elh);
  EdgeListIterator_T* eli = static_cast<EdgeListIterator_T*>(elih);
  eli->ptr++;
  adjust_edgelist_iter(el, eli);
  return eli;
}

bool has_next_edge_iter(const EdgeList elh, const EdgeListIterator elih) {
  EdgeList_T* el = static_cast<EdgeList_T*>(elh);
  EdgeListIterator_T* eli = static_cast<EdgeListIterator_T*>(elih);
  return eli->vli != el->vl->end_value();
}

Edge get_edge_from_iter(const EdgeList elh, const EdgeListIterator elih) {
  EdgeList_T* el = static_cast<EdgeList_T*>(elh);
  EdgeListIterator_T* eli = static_cast<EdgeListIterator_T*>(elih);
  Vertex v = get_vertex_from_iter(el->vl, eli->vli);

  if (eli->d == Direction::IN) {
    Edge_T* e = new Edge_T(eli->ptr->get_neighbor().GetValue(), v,
                           eli->ptr->get_data());
    return e;
  } else {
    Edge e = new Edge_T(v, eli->ptr->get_neighbor().GetValue(),
                        eli->ptr->get_data());
    return e;
  }
}

#endif
