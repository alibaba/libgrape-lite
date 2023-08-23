#!/usr/bin/env python3
import sys
import os

source = int(sys.argv[2])
n = int(sys.argv[3])

vset = set()
edges = list()

path = sys.argv[1]
prefix = os.path.dirname(path)
filename = os.path.basename(path)
ext = ''

with open(path, 'r') as fi:
    for line in fi:
        edges.append(line)

print("Graph loaded")

for x in range(2, 6 + 1, 2):
    dataset_prefix = prefix + "/" + str(x)
    if not os.path.exists(dataset_prefix):
        os.mkdir(dataset_prefix)

    with open(dataset_prefix + "/" + filename + ".v", 'w') as fo:
        for curr_x in range(0, x):
            for v in range(0, n):
                fo.write("%d\n" % (v + n * curr_x))

    with open(dataset_prefix + "/" + filename, 'w') as fo:
        for curr_x in range(0, x):
            for line in edges:
                parts = line.split()
                op = parts[0]
                if op == "a" or op == "d":
                    u = int(parts[1])
                    v = int(parts[2])
                    weight = None
                    if len(parts) == 4:
                        weight = parts[3]
                    if weight is None:
                        fo.write("%s %d %d\n" % (op, u + n * curr_x, v + n * curr_x))
                    else:
                        fo.write("%s %d %d %s\n" % (op, u + n * curr_x, v + n * curr_x, weight))
                else:
                    u = int(parts[0])
                    v = int(parts[1])
                    weight = None
                    if len(parts) == 3:
                        weight = parts[2]
                    if weight is None:
                        fo.write("%d %d\n" % (u + n * curr_x, v + n * curr_x))
                    else:
                        fo.write("%d %d %s\n" % (u + n * curr_x, v + n * curr_x, weight))
            if curr_x != 0 and not (op == "a" or op == "d"):
                if weight is not None:
                    fo.write("%d %d 1\n" % (source, source + curr_x * n))
                else:
                    fo.write("%d %d\n" % (source, source + curr_x * n))
