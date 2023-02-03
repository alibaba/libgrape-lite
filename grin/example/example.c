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

#include "grin/include/topology/structure.h"
#include "grin/include/topology/vertexlist.h"
#include "grin/include/topology/adjacentlist.h"
#include "grin/include/topology/edgelist.h"
#include "grin/include/partition/partition.h"

int main(int argc, char* argv[]) {
    // travere graph
    void* g;
    void* vl = get_vertex_list(g);
    void* vli = get_vertex_list_begin(vl);
    while (has_next_vertex_iter(vl, vli)) {
        void* v = get_vertex_from_iter(vl, vli);
        void* al = get_adjacent_list(g, OUT, v);
        void* ali = get_adjacent_list_begin(al);
        while (has_next_adjacent_iter(al, ali)) {
            void* u = get_neighbor_from_iter(al, ali);
            void* w = get_adjacent_edge_data_value(al, ali);
            DataType vdt = get_vertex_id_data_type(g);
            DataType wdt = get_adjacent_edge_data_type(g);
            ali = get_next_adjacent_iter(al, ali);
        }
        vli = get_next_vertex_iter(vl, vli);
    }
    return 0;
}