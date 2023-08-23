#!/usr/bin/env python3
import sys

vset = set()
for path in sys.argv[1:-1]:
    with open(path, 'r') as fi:
        for line in fi:
            parts = line.split()
            if len(parts) >= 2:
                src = int(parts[0])
                dst = int(parts[1])
                vset.add(src)
                vset.add(dst)

with open(sys.argv[-1], 'w') as fo:
    for v in vset:
        fo.write('%d\n' % v)
