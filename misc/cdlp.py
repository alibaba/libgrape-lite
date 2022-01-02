#!/usr/bin/env python3
import networkx as nx
import sys

if len(sys.argv) != 3:
    sys.exit("Usage: [input_graph] [input_result]")

f_graph = sys.argv[1]
f_result = sys.argv[2]

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

print("Comparing result with " + f_result)

comms = set()
with open(f_result, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) > 0:
            arr = line.split()
            v = int(arr[0])
            comm = int(arr[1])
            comms.add(comm)
if len(comms) != 11019:
    sys.exit("Wrong answer, # of comm should be 11019 but %d is produced" % len(comms))
