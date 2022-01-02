#!/usr/bin/env python3

import random
import sys

if len(sys.argv) != 2:
    print('Usage: python associate_weights.py file_name.mtx')
    sys.exit()

with open(sys.argv[1], 'r') as fi:
    with open(sys.argv[1].split('.')[0] + '.random.weight.mtx', 'w') as fo:
        is_edge = False
        for line in fi:
            if line[0] == '%' or line == '\n':
                fo.write(line)
            elif not is_edge:
                is_edge = True
                fo.write(line)
            elif is_edge:
                line = line.strip()
                arr = line.split()
                fo.write('%s %s %d\n' % (arr[0], arr[1], random.randint(1, 63)))
