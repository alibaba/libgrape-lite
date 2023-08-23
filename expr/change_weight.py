#!/usr/bin/env python3
import random
import sys

raw_path = sys.argv[1]
percentage = 0.01

edges = []
with open(raw_path, 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) > 0 and line[0] != '#' and line[0] != '%' and line != '\n':
            edges.append(line)

n_edges = len(edges)
changed_n_edges = int(percentage * n_edges)
used_eids = set()

changed_edges = []

begin_eid = int(0.98 * n_edges)

random.seed(0)
for _ in range(changed_n_edges):
    eid = random.randint(begin_eid, n_edges - 1)
    while eid in used_eids:
        eid = random.randint(begin_eid, n_edges - 1)

    used_eids.add(eid)
    line = edges[eid]
    arr = line.split()
    src = arr[0]
    dst = arr[1]
    weight = int(arr[2])
    new_weight = random.randint(1, 64)
    while weight == new_weight:
        new_weight = random.randint(1, 64)

    changed_edges.append("d %s %s 0" % (src, dst))
    changed_edges.append("a %s %s %d" % (src, dst, new_weight))
    edges[eid] = "%s %s %d" % (src, dst, new_weight)

with open(raw_path + ".cw." + str(percentage) + ".update", 'w') as fo:
    for e in changed_edges:
        fo.write(e + "\n")

with open(raw_path + ".cw." + str(percentage) + ".updated", 'w') as fo:
    for e in edges:
        fo.write(e + "\n")
