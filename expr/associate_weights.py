#!/usr/bin/env python3

"""
Simple python script to generate random weight values for file_name.mtx graph
"""

import random
import fileinput
import sys

### check command line args
if len(sys.argv) != 3:
    print('Usage: python associate_weights.py file_name.mtx')
    sys.exit()

### Associate random weight (0, 63) values for each edge in the graph
ograph = open(sys.argv[2], 'w')
for line in fileinput.input(sys.argv[1]):
    if line[0] == '#' or line[0] == '%' or line == '\n':
        continue
    else:
        line = line.split('\n')
        new_line = line[0] + ' ' + str(random.randint(1, 64)) + '\n'
        ograph.write(new_line)

ograph.close()
