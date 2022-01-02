#!/usr/bin/env python3
import networkx as nx
import sys

if len(sys.argv) != 3:
    sys.exit("Usage: [input_graph] [input_result]")

f_graph = sys.argv[1]
f_result = sys.argv[2]

g = nx.DiGraph()

print("Loading graph " + f_graph)
with open(f_graph, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) == 0 or line[0] in ('%', '#'):
            continue
        tmp = line.split()
        src = int(tmp[0])
        dst = int(tmp[1])

        g.add_edge(src, dst)

print("Evaluating WCC")
cc_sets = nx.weakly_connected_components(g)

correct_ans = {}
cc_id = 0
for cc_set in cc_sets:
    for oid in cc_set:
        correct_ans[oid] = cc_id
    cc_id += 1

print("Comparing result with " + f_result)

ans = []
with open(f_result, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) > 0:
            arr = line.split()
            v = int(arr[0])
            cc_id = int(arr[1])
            ans.append((v, cc_id))

cluster_name1 = dict()
cluster_name2 = dict()

for oid, cc_id1 in ans:
    cc_id2 = correct_ans[oid]
    if cc_id1 not in cluster_name1 and cc_id2 not in cluster_name2:
        curr_size = len(cluster_name1)
        cluster_name1[cc_id1] = curr_size
        cluster_name2[cc_id2] = curr_size
    elif cc_id1 in cluster_name1 and cc_id2 in cluster_name2:
        if cluster_name1[cc_id1] != cluster_name2[cc_id2]:
            sys.exit("Error result: v = %d %d vs %d" % (oid, cluster_name1[cc_id1], cluster_name2[cc_id2]))
    else:
        sys.exit("Error result: v = %d" % oid)
