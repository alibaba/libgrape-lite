#!/usr/bin/env python3
import os
import sys
import random


def pop_first(d: dict):
    for k in d:
        return k, d.pop(k)


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python gen.py [path] [name] [-header]")

    path = sys.argv[1]
    name = sys.argv[2]
    header = False
    if len(sys.argv) == 4:
        if sys.argv[3] != '-header':
            sys.exit("Illegal arg: " + sys.argv[3])
        else:
            header = True

    prefix = os.path.dirname(path)
    filename = os.path.basename(path)
    base = filename
    ext = ''
    random.seed(10)

    if '.' in filename:
        base = os.path.splitext(filename)[0]
        ext = '.'.join(os.path.splitext(filename)[1:])

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

    add_size = 288542 / 2
    del_size = 288542 / 2

    print("Vertex num: ", vertex_num)
    print("Add size: ", add_size)
    print("Del size: ", del_size)

    if not os.path.exists('%s/%s' % (prefix, name)):
        os.mkdir('%s/%s' % (prefix, name))

    base += '_w'
    fp_vm = open('%s/%s/%s.v' % (prefix, name, base), 'w')
    fp_base = open('%s/%s/%s.base' % (prefix, name, base), 'w')
    fp_append = open('%s/%s/%s.update' % (prefix, name, base), 'w')
    fp_updated = open('%s/%s/%s.updated' % (prefix, name, base), 'w')

    G = {}

    # Load graph
    print("Loading graph")
    num_lines = 0
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()

            if len(line) == 0 or line.startswith('#') or line.startswith('%'):
                continue
            num_lines += 1
            if header and num_lines == 1:
                continue
            parts = line.split()
            u_gid = vm[int(parts[0])]
            v_gid = vm[int(parts[1])]
            # skip self cycle
            if u_gid == v_gid:
                continue
            if u_gid not in G:
                G[u_gid] = dict()
            oes = G[u_gid]

            weight = random.randint(1, 64)
            oes[v_gid] = weight

    print("Compensating degree")
    # Compensate 0-degree
    for u_gid in range(vertex_num):
        if u_gid not in G:
            G[u_gid] = dict()
            oes = G[u_gid]
            v_gid = random.randrange(0, vertex_num)
            while v_gid == u_gid:
                v_gid = random.randrange(0, vertex_num)
            weight = random.randint(1, 64)
            oes[v_gid] = weight

    print("Generating add")
    processed_edges = set()
    for u_gid in G:
        oes = G[u_gid]

        need_add = random.randint(0, 1)
        if need_add == 1 and len(oes) > 1 and add_size > 0:
            v_gid, weight = pop_first(oes)

            fp_append.write('a %d %d %s\n' % (u_gid, v_gid, weight))
            fp_updated.write('%d %d %s\n' % (u_gid, v_gid, weight))
            processed_edges.add((u_gid, v_gid))
            add_size -= 1

        for v_gid in oes:
            weight = oes[v_gid]
            fp_base.write('%d %d %s\n' % (u_gid, v_gid, weight))
            fp_updated.write('%d %d %s\n' % (u_gid, v_gid, weight))

    print("Generating del")
    while del_size > 0:
        u_gid = random.randrange(0, vertex_num)
        v_gid = random.randrange(0, vertex_num)
        e = (u_gid, v_gid)
        oes = G[u_gid]

        if e in processed_edges or v_gid in oes:
            continue
        del_size -= 1
        processed_edges.add(e)
        weight = random.randint(1, 64)
        fp_base.write('%d %d %d\n' % (u_gid, v_gid, weight))
        fp_append.write('d %d %d 0\n' % (u_gid, v_gid))

        oes[v_gid] = weight

    print("Writing vm")
    for v in range(vertex_num):
        fp_vm.write('%d\n' % v)

    fp_vm.close()
    fp_base.close()
    fp_append.close()
    fp_updated.close()


if __name__ == '__main__':
    main()
