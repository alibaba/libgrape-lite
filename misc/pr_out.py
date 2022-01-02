#!/usr/bin/env python3
import networkx as nx
import sys
from networkx.algorithms.link_analysis import pagerank

if len(sys.argv) != 3:
    sys.exit("Usage: [input_graph] [directed/undirected]")

d = 0.8
f_graph = sys.argv[1]
dir = sys.argv[2]

if dir == "directed":
    g = nx.DiGraph()
elif dir == "undirected":
    g = nx.Graph()
else:
    sys.exit("Invalid arg: " + dir)

with open(f_graph, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) == 0 or line[0] in ('%', '#'):
            continue
        tmp = line.split()
        src = int(tmp[0])
        dst = int(tmp[1])

        g.add_edge(src, dst)

correct_ans = pagerank(g, alpha=d)
for v in correct_ans:
    print("%d %f" % (v, correct_ans[v]))

print(sum(correct_ans.values()))

value = [0 for v in g]
delta = [(1 - d) for v in g]
n_non_dangling = 0

for v in g:
    if len(g[v]) > 0:
        n_non_dangling += 1

# for i in range(40):
#     total_dangling = 0
#
#     for u in g:
#         old_delta = delta[u]
#         delta[u] = 0
#         value[u] += old_delta
#         if len(g[u]) == 0:
#             total_dangling += old_delta
#         for v in g[u]:
#             delta[v] += d * old_delta / len(g[u])
#
#     for v in g:
#         delta[v] += d * total_dangling / len(g)

for i in range(40):
    total_dangling = 0

    for u in g:
        if len(g[u]) > 0:
            old_delta = delta[u]
            delta[u] = 0
            value[u] += old_delta
            for v in g[u]:
                delta[v] += d * old_delta / len(g[u])
    for u in g:
        if len(g[u]) == 0:
            old_delta = delta[u]
            delta[u] = 0
            value[u] += old_delta
            total_dangling += old_delta

    for v in g:
        delta[v] += d * total_dangling / len(g)

for v in g:
    value[v] /= len(g)
print(value)
print(sum(value))
