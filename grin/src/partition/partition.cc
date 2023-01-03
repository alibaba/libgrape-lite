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

// TODO: implement partition.cc

#ifndef GRIN_PARTITION_PARTITION_H_
#define GRIN_PARTITION_PARTITION_H_

#include "grin/src/predefine.h"
#include <string>
#include <cstdlib>

#ifdef PARTITION_STRATEGY
// basic partition informations
size_t get_total_partitions_number(const PartitionedGraph g) {
    return g->fnum();
}

PartitionList get_local_partitions(const PartitionedGraph g) {
    unsigned fid = g->fid();
    return &fid;
}

size_t get_partition_list_size(const PartitionedGraph g) {
    return 1;
}

Partition get_partition_from_list(const PartitionList pl, const size_t idx) {
    return *(pl + idx);
}

//PartitionList create_partition_list();

//void destroy_partition_list(PartitionList);

//bool insert_partition_to_list(PartitionList, const Partition);

void* get_partition_info(const Partition p) {
    return (void*)(&p);
}

Graph get_local_graph_from_partition(const PartitionedGraph g, const Partition p) {
    if (g->fid() != p) return NULL_GRAPH;
    return g;
}

// serialization & deserialization
char* serialize_remote_partition(const PartitionedGraph g, const RemotePartition rp) {
    char out[12];
    sprintf(out, "%s", std::to_string(rp).c_str());
    return out;
}

char* serialize_remote_vertex(const PartitionedGraph g, const RemoteVertex rv) {
    VID_T gid = g->Vertex2Gid(rv);
    char out[12];
    sprintf(out, "%s", std::to_string(gid).c_str());
    return out;
}

char* serialize_remote_edge(const PartitionedGraph g, const RemoteEdge re) {
    VID_T src = g->Vertex2Gid(Vertex(re.src));
    VID_T dst = g->Vertex2Gid(Vertex(re.dst));
    char out[38];
    sprintf(out, "%s %s %s", std::to_string(src).c_str(), std::to_string(dst).c_str(), std::to_string(re.edata).c_str());
    return out;
}

RemotePartition get_partition_from_deserialization(const PartitionedGraph g,
                                                   const char* msg) {
    RemotePartition p;
    std::stringstream ss(msg);
    ss >> p;
    return p;                 
}

RemoteVertex get_vertex_from_deserialization(const PartitionedGraph g,
                                             const char* msg) {
    VID_T v;
    std::stringstream ss(msg);
    ss >> v;
    return Vertex(v);
}

RemoteEdge get_edge_from_deserialization(const PartitionedGraph g,
                                         const char* msg) {
    VID_T src, dst;
    EDATA_T data;
    std::stringstream ss(msg);
    ss >> src >> dst >> data;
    return Edge(src, dst, data);
}

// For local vertex: could get its properties locally, but not sure to have all
// its edges locally, which depends on the partition strategy; every vertex
// could be local in 1~n partitions.
bool is_local_vertex(const PartitionedGraph g, const Partition p, const Vertex v) {
    return g->IsInnerVertex(v);
}

// For local edge: could get its properties locally;
// every edge could be local in 1/2/n partitions
bool is_local_edge(const PartitionedGraph g, const Partition p, const Edge e) {
    return g->IsInnerVertex(grape::Vertex<VID_T>(e.src)) || g->IsInnerVertex(grape::Vertex<VID_T>(e.dst));
}

// For a non-local vertex/edge, get its master partition (a remote partition);
// also, we can get its master vertex/edge (a remote vertex/edge);
// for a local vertex/edge, an invalid value would be returned
RemotePartition get_master_partition_for_vertex(const PartitionedGraph g,
                                                const Partition p, const Vertex v) {
    RemotePartition rp = g->GetFragId(v);
    if (rp == p) {
        return NULL_PARTITION;
    }
    return rp;
}

RemotePartition get_master_partition_for_edge(const PartitionedGraph g,
                                              const Partition p, const Edge e) {
    RemotePartition src_rp = g->GetFragId(grape::Vertex<VID_T>(e.src));
    RemotePartition dst_rp = g->GetFragId(grape::Vertex<VID_T>(e.dst));
    if (src_rp == p) {
        if (dst_rp == p) {
            return NULL_PARTITION;
        } else {
            return dst_rp;
        }
    } else {
        if (dst_rp == p) {
            return src_rp;
        }
        else {
            return NULL_PARTITION;
        }
    }
    return NULL_PARTITION;
}

RemoteVertex get_master_vertex_for_vertex(const PartitionedGraph g,
                                          const Partition p, const Vertex v) {
    return RemoteVertex(g->Vertex2Gid(v));
}

RemoteEdge get_master_edge_for_edge(const PartitionedGraph g, const Partition p,
                                    const Edge e) {
    return RemoteEdge(g->Vertex2Gid(Vertex(e.src)),
                      g->Vertex2Gid(Vertex(e.dst)),
                      e.edata);
}

// get the partitions in which a vertex exists
RemotePartitionList get_all_partitions_for_vertex(const PartitionedGraph g,
                                                  const Partition p,
                                                  const Vertex v) {
    unsigned fid = g->GetFragId(v);
    return &fid;
}

size_t get_remote_partition_list_size(const PartitionedGraph g,
                                      const Partition p,
                                      const Vertex v) {
    return 1;
}

RemotePartition get_remote_partition_from_list(const RemotePartitionList rpl,
                                               const size_t idx) {
    return *(rpl + idx);
}

//RemotePartitionList create_remote_partition_list();

//void destroy_remote_partition_list(RemotePartitionList);

//bool insert_remote_partition_to_list(RemotePartitionList,
//                                     const RemotePartition);

// get the replicas of a vertex
RemoteVertexList get_all_replicas_for_vertex(const PartitionedGraph g,
                                             const Partition p, const Vertex v) {
    return NULL;
}

size_t get_remote_vertex_list_size(const PartitionedGraph g,
                                   const Partition p, const Vertex v) {
    return 0;
}

RemoteVertex get_remote_vertex_from_list(const RemoteVertexList rvl, const size_t idx) {
    return *(rvl + idx);
};

//RemoteVertexList create_remote_vertex_list();

//void destroy_remote_vertex_list(RemoteVertexList);

//bool insert_remote_vertex_to_list(RemoteVertexList, const RemoteVertex);
#endif

#if defined(PARTITION_STRATEGY) && defined(ENABLE_VERTEX_LIST)
VertexList get_local_vertices(const PartitionedGraph g, const Partition p) {
    grape::VertexRange<VID_T> vr = g->InnerVertices();
    return vr;
}

VertexList get_non_local_vertices(const PartitionedGraph g, const Partition p) {
    grape::VertexRange<VID_T> vr = g->OuterVertices();
    return vr;
}
#endif

#endif  // GRIN_PARTITION_PARTITION_H_
