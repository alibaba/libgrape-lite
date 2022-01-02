#!/usr/bin/env python3
import networkx as nx
import sys

if len(sys.argv) != 3:
    sys.exit("Usage: [input_graph] [input_result]")

f_graph = sys.argv[1]
f_result = sys.argv[2]
lcc_tol = 0.00001

g = nx.Graph()

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

print("Evaluating LCC")
correct_ans = nx.clustering(g)

print("Comparing result with " + f_result)

ans = dict()
with open(f_result, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) > 0:
            arr = line.split()
            v = int(arr[0])
            pr = float(arr[1])
            ans[v] = pr

if len(ans) != len(correct_ans):
    sys.exit("Wrong number of vertices")

for v in correct_ans:
    lcc1 = correct_ans[v]
    lcc2 = ans[v]
    if abs(lcc1 - lcc2) >= lcc_tol:
        sys.exit("Error result: (v, rank) = (%d, %f). should be %f" % (v, lcc2, lcc1))
