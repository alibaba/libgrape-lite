#!/usr/bin/env python3
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import sys

if len(sys.argv) != 6:
    sys.exit("Usage: [input_graph] [input_result] [directed/undirected] [source] [sssp/bfs]")

f_graph = sys.argv[1]
f_result = sys.argv[2]
dir = sys.argv[3]
source = int(sys.argv[4])
algo = sys.argv[5]

if dir == "directed":
    g = nx.DiGraph()
elif dir == "undirected":
    g = nx.Graph()
else:
    sys.exit("Invalid arg: " + dir)

print("Loading graph " + f_graph)
with open(f_graph, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) == 0 or line[0] in ('%', '#'):
            continue
        tmp = line.split()
        src = int(tmp[0])
        dst = int(tmp[1])
        weight = int(tmp[2])

        g.add_edge(src, dst, data=weight)

print("Evaluating " + algo)

if algo == "sssp":
    correct_ans = shortest_path_length(g, source=source, weight="data")
elif algo == "bfs":
    correct_ans = shortest_path_length(g, source=source, weight=None)
else:
    sys.exit("Unsupported algo: " + algo)

print("Comparing result with " + f_result)

ans = dict()
with open(f_result, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) > 0:
            arr = line.split()
            v = int(arr[0])
            dist = float(arr[1])
            ans[v] = dist

matched_n_vertices = 0
for v in correct_ans:
    dist1 = correct_ans[v]
    dist2 = ans[v]
    matched_n_vertices += 1
    if dist1 != dist2:
        sys.exit("Error result: (v, result) = (%d, %f). should be %f" % (v, dist2, dist1))

if matched_n_vertices != len(correct_ans):
    sys.exit("Wrong number of vertices %d vs %d" % (matched_n_vertices, len(correct_ans)))
