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
#include "grin/include/partition/partition.h"

#include <string>
#include <cstdlib>

#ifdef PARTITION_STRATEGY
// basic partition informations
size_t get_total_partitions_number(const PartitionedGraph pgh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    return pg->fnum();
}

size_t get_total_vertices_number(const PartitionedGraph pgh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    return pg->GetTotalVerticesNum();
}

PartitionList get_local_partitions(const PartitionedGraph pgh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    PartitionList pl = new Partition[1];
    pl[0] = pg->fid();
    return pl;
}

size_t get_partition_list_size(const PartitionList pl) {
    return 1;
} 

Partition get_partition_from_list(const PartitionList pl, const size_t idx) {
    return *(pl + idx);
}

//PartitionList create_partition_list();

//void destroy_partition_list(PartitionList);

//bool insert_partition_to_list(PartitionList, const Partition);

void* get_partition_info(const Partition p) {
    return NULL;
}

Graph get_local_graph_from_partition(const PartitionedGraph pgh, const Partition p) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (pg->fid() != p) return NULL_GRAPH;
    return pg;
}

// serialization & deserialization
char* serialize_remote_partition(const PartitionedGraph pgh, const RemotePartition rp) {
    char* out = new char[12];
    snprintf(out, 12, "%s", std::to_string(rp).c_str());
    return out;
}

char* serialize_remote_vertex(const PartitionedGraph pgh, const RemoteVertex rv) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    G_VID_T gid = pg->Vertex2Gid(Vertex_G(rv));
    char* out = new char[12];
    snprintf(out, 12, "%s", std::to_string(gid).c_str());
    return out;
}

char* serialize_remote_edge(const PartitionedGraph pgh, const RemoteEdge reh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    RemoteEdge_T* re = static_cast<RemoteEdge_T*>(reh);
    G_VID_T src = pg->Vertex2Gid(Vertex_G(re->src));
    G_VID_T dst = pg->Vertex2Gid(Vertex_G(re->dst));
    char* out = new char[38];
    snprintf(out, 38, "%s %s %s", std::to_string(src).c_str(), std::to_string(dst).c_str(), std::to_string(re->edata).c_str());
    return out;
}

Partition get_partition_from_deserialization(const PartitionedGraph pgh,
                                             const char* msg) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    Partition p;
    std::stringstream ss(msg);
    ss >> p;
    if (p >= pg->fnum()) {
        return NULL_PARTITION;
    }
    return p;                 
}

Vertex get_vertex_from_deserialization(const PartitionedGraph pgh,
                                       const Partition p,
                                       const char* msg) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    G_VID_T gv;
    std::stringstream ss(msg);
    ss >> gv;
    Vertex_G v;
    if (!pg->Gid2Vertex(gv, v)) {
        return NULL_VERTEX;
    }
    return v.GetValue();
}

Edge get_edge_from_deserialization(const PartitionedGraph pgh,
                                   const Partition p,
                                   const char* msg) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    G_VID_T src, dst;
    G_EDATA_T data;
    std::stringstream ss(msg);
    ss >> src >> dst >> data;
    Vertex_G v1, v2;
    if (!pg->Gid2Vertex(src, v1) || !pg->Gid2Vertex(dst, v2)) {
        return NULL_EDGE;
    }
    Edge_T* e = new Edge_T(v1.GetValue(), v2.GetValue(), data);
    return e;
}

// For local vertex: could get its properties locally, but not sure to have all
// its edges locally, which depends on the partition strategy; every vertex
// could be local in 1~n partitions.
bool is_local_vertex(const PartitionedGraph pgh, const Partition p, const Vertex v) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    return pg->IsInnerVertex(Vertex_G(v));
}

// For local edge: could get its properties locally;
// every edge could be local in 1/2/n partitions
bool is_local_edge(const PartitionedGraph pgh, const Partition p, const Edge eh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    Edge_T* e = static_cast<Edge_T*>(eh);
    return pg->IsInnerVertex(Vertex_G(e->src)) || pg->IsInnerVertex(Vertex_G(e->dst));
}

// For a non-local vertex/edge, get its master partition (a remote partition);
// also, we can get its master vertex/edge (a remote vertex/edge);
// for a local vertex/edge, an invalid value would be returned
RemotePartition get_master_partition_for_vertex(const PartitionedGraph pgh,
                                                const Partition p, const Vertex v) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    RemotePartition rp = pg->GetFragId(Vertex_G(v));
    if (rp == p) {
        return NULL_PARTITION;
    }
    return rp;
}

RemotePartition get_master_partition_for_edge(const PartitionedGraph pgh,
                                              const Partition p, const Edge eh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    Edge_T* e = static_cast<Edge_T*>(eh);
    RemotePartition src_rp = pg->GetFragId(Vertex_G(e->src));
    RemotePartition dst_rp = pg->GetFragId(Vertex_G(e->dst));
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

RemoteVertex get_master_vertex_for_vertex(const PartitionedGraph pgh,
                                          const Partition p, const Vertex v) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    return pg->Vertex2Gid(Vertex_G(v));
}

RemoteEdge get_master_edge_for_edge(const PartitionedGraph pgh, const Partition p,
                                    const Edge eh) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    Edge_T* e = static_cast<Edge_T*>(eh);
    RemoteEdge_T* re = new RemoteEdge_T(pg->Vertex2Gid(Vertex_G(e->src)),
                                        pg->Vertex2Gid(Vertex_G(e->dst)),
                                        e->edata);
    return re;
}

// get the partitions in which a vertex exists
RemotePartitionList get_remote_partition_list_for_vertex(const PartitionedGraph pgh,
                                                         const Partition p,
                                                         const Vertex v) {
    if (is_local_vertex(pgh, p, v)) {
        return NULL_LIST;
    }
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    RemotePartitionList rpl = new RemotePartition[1];
    rpl[0] = pg->GetFragId(Vertex_G(v));
    return rpl;
}

size_t get_remote_partition_list_size(const RemotePartitionList rpl) {
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
RemoteVertexList get_all_replicas_for_vertex(const PartitionedGraph pgh,
                                             const Partition p, const Vertex v) {
    return NULL_LIST;
}

size_t get_remote_vertex_list_size(const RemoteVertexList rvl) {
    return 0;
}

RemoteVertex get_remote_vertex_from_list(const RemoteVertexList rvl, const size_t idx) {
    return NULL_REMOTE_VERTEX;
};

//RemoteVertexList create_remote_vertex_list();

//void destroy_remote_vertex_list(RemoteVertexList);

//bool insert_remote_vertex_to_list(RemoteVertexList, const RemoteVertex);
#endif

#if defined(PARTITION_STRATEGY) && defined(ENABLE_VERTEX_LIST)
VertexList get_local_vertices(const PartitionedGraph pgh, const Partition p) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (pg->fid() != p) {
        return NULL_LIST;
    }
    VertexList_T* vl = new VertexList_T(pg->InnerVertices());
    return vl;
}

VertexList get_remote_vertices(const PartitionedGraph pgh, const Partition p) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (pg->fid() != p) {
        return NULL_LIST;
    }
    VertexList_T* vl = new VertexList_T(pg->OuterVertices());
    return vl;
}

VertexList get_remote_vertices_by_partition(const PartitionedGraph pgh, const RemotePartition p) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (pg->fid() == p) {
        return NULL_LIST;
    }
    VertexList_T* vl = new VertexList_T(pg->OuterVertices(p));
    return vl;
}

#endif

#if defined(PARTITION_STRATEGY) && defined(ENABLE_ADJACENT_LIST)
AdjacentList get_local_adjacent_list(const PartitionedGraph pgh, const Direction d, const Partition p, const Vertex v) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (d == Direction::BOTH || pg->fid() != p) {
        return NULL_LIST;
    } else if (d == Direction::IN) { 
        AdjacentList_T* al = new AdjacentList_T(pg->GetIncomingInnerVertexAdjList(Vertex_G(v)));
        return al;
    } else {
        AdjacentList_T* al = new AdjacentList_T(pg->GetOutgoingInnerVertexAdjList(Vertex_G(v)));
        return al;
    }
}

AdjacentList get_remote_adjacent_list(const PartitionedGraph pgh, const Direction d, const Partition p, const Vertex v) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (d == Direction::BOTH || pg->fid() != p) {
        return NULL;
    } else if (d == Direction::IN) { 
        AdjacentList_T* al = new AdjacentList_T(pg->GetIncomingOuterVertexAdjList(Vertex_G(v)));
        return al;
    } else {
        AdjacentList_T* al = new AdjacentList_T(pg->GetOutgoingOuterVertexAdjList(Vertex_G(v)));
        return al;
    }
}

AdjacentList get_remote_adjacent_list_by_partition(const PartitionedGraph pgh, const Direction d, const RemotePartition p, const Vertex v) {
    PartitionedGraph_T* pg = static_cast<PartitionedGraph_T*>(pgh);
    if (d == Direction::BOTH) {
        return NULL;
    } else if (pg->fid() == p) {
        return get_local_adjacent_list(pgh, d, p, v);
    } else if (d == Direction::IN) {
        AdjacentList_T* al = new AdjacentList_T(pg->GetIncomingAdjList(Vertex_G(v), p));
        return al;
    } else {
        AdjacentList_T* al = new AdjacentList_T(pg->GetOutgoingAdjList(Vertex_G(v), p));
        return al;
    }
}

#endif

