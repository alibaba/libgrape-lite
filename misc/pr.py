#!/usr/bin/env python3
import networkx as nx
import sys

if len(sys.argv) != 4:
    sys.exit("Usage: [input_graph] [input_result] [directed/undirected]")

f_graph = sys.argv[1]
f_result = sys.argv[2]
dir = sys.argv[3]

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

        g.add_edge(src, dst)

print("Evaluating pagerank")


def pagerank(
        G,
        alpha=0.85,
        max_iter=100,
        weight="weight",
):
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    x = dict.fromkeys(W, 1.0 / N)
    p = dict.fromkeys(W, 1.0 / N)

    # Use personalization vector if dangling vector not specified
    dangling_weights = p
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
    return x


correct_ans = pagerank(g, alpha=0.85, max_iter=30)

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
    sys.exit("Wrong number of vertices %d vs %d" % (len(ans), len(correct_ans)))

for v in correct_ans:
    pr1 = correct_ans[v]
    pr2 = ans[v]
    diff = abs(pr1 - pr2)
    # Because the GPU program uses a float type but this program for verification uses double.
    # So it's fair for having 2% difference
    if diff / pr1 > 0.02:
        print("Diff percentage: " + str(diff / pr1))
        sys.exit("Error result: (v, rank) = (%d, %f). should be %f" % (v, pr2, pr1))
