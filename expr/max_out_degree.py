#!/usr/bin/env python3
import sys
import operator

od = {}
top = 10

with open(sys.argv[1], 'r') as fi:
    for line in fi:
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split()
        src = int(parts[0])
        if src not in od:
            od[src] = 0
        od[src] += 1

v_out = sorted(od.items(), key=operator.itemgetter(1), reverse=True)

print("node    outdegree")
for k in v_out:
    print(str(k[0]) + ' ' + str(k[1]))
    top -= 1
    if top <= 0:
        break