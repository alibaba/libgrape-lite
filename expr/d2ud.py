#!/usr/bin/env python3
import sys
import os

path = sys.argv[1]
prefix = os.path.dirname(path)
filename = os.path.basename(path)
base = filename
ext = ''

if '.' in filename:
    base = os.path.splitext(filename)[0]
    ext = '.'.join(os.path.splitext(filename)[1:])

G = {}
print("Loading " + path)
with open(path, 'r') as fi:
    for line in fi:
        line = line.strip()

        if len(line) == 0 or line.startswith('#') or line.startswith('%'):
            continue

        parts = line.split()
        op = parts[0][0]
        if op == 'a' or op == 'd':
            u = int(parts[1])
            v = int(parts[2])
            weight = None
            if len(parts) == 4:
                weight = parts[3]
            if u not in G:
                G[u] = {}
            G[u][v] = (op, weight)
            if v not in G:
                G[v] = {}
            if u not in G[v]:
                G[v][u] = (op, weight)
        else:
            u = int(parts[0])
            v = int(parts[1])
            weight = None
            if len(parts) == 3:
                weight = parts[2]
            if u not in G:
                G[u] = {}
            G[u][v] = weight
            if v not in G:
                G[v] = {}
            if u not in G[v]:
                G[v][u] = weight

with open("{PREFIX}/{NAME}_ud{EXT}".format(PREFIX=prefix, NAME=base, EXT=ext), 'w') as fo:
    for u in G:
        oes = G[u]
        for v in oes:
            weight = oes[v]

            if type(weight) is tuple:
                op, weight = weight

                if weight is not None:
                    fo.write('%s %d %d %s\n' % (op, u, v, weight))
                else:
                    fo.write('%s %d %d\n' % (op, u, v))
            else:
                if weight is not None:
                    fo.write('%d %d %s\n' % (u, v, weight))
                else:
                    fo.write('%d %d\n' % (u, v))
