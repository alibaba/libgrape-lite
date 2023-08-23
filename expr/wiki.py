#!/usr/bin/env bash
import os
import sys
import random

G = {}

path = sys.argv[1]
prefix = os.path.dirname(path)
filename = os.path.basename(path)
base = filename
ext = ''

header = False
num_lines = 0
vm = {}
vertex_num = 0

# create vm
print("Creating VM")
with open(path, 'r') as fi:
    for line in fi:
        line = line.strip()

        if len(line) == 0 or line.startswith('#') or line.startswith('%'):
            continue
        num_lines += 1
        if header and num_lines == 1:
            continue
        parts = line.split()
        u = int(parts[0])
        v = int(parts[1])

        if u not in vm:
            vm[u] = vertex_num
            vertex_num += 1
        if v not in vm:
            vm[v] = vertex_num
            vertex_num += 1

with open(path, 'r') as fi:
    for line in fi:
        line = line.strip()

        if len(line) == 0 or line.startswith('#') or line.startswith('%'):
            continue

        parts = line.split()
        src_gid = vm[int(parts[0])]
        dst_gid = vm[int(parts[1])]
        op = int(parts[2])
        if op == 2:
            weight = random.randint(1, 64)
            if src_gid not in G:
                G[src_gid] = {}
            G[src_gid][dst_gid] = weight

with open(prefix + "/" + filename + "_w.base", 'w') as fo:
    for u in G:
        oes = G[u]
        for v in oes:
            weight = oes[v]
            fo.write('%d %d %d\n' % (u, v, weight))

with open(prefix + "/" + filename + "_w.update", 'w') as fo:
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()

            if len(line) == 0 or line.startswith('#') or line.startswith('%'):
                continue

            parts = line.split()
            src_gid = vm[int(parts[0])]
            dst_gid = vm[int(parts[1])]
            op = int(parts[2])

            if src_gid not in vm or dst_gid not in vm:
                continue
            if op == 1:
                fo.write("d %d %d 0\n" % (src_gid, dst_gid))
                if src_gid in G and dst_gid in G[src_gid]:
                    G[src_gid].pop(dst_gid)
            elif op == 4:
                weight = random.randint(1, 64)
                fo.write("a %d %d %d\n" % (src_gid, dst_gid, weight))
                if src_gid not in G:
                    G[src_gid] = {}
                G[src_gid][dst_gid] = weight

with open(prefix + "/" + filename + "_w.updated", 'w') as fo:
    for u in G:
        oes = G[u]
        for v in oes:
            weight = oes[v]
            fo.write("%d %d %d\n" % (u, v, weight))

with open(prefix + "/" + filename + ".v", 'w') as fo:
    for v in range(vertex_num):
        fo.write('%d\n' % v)
