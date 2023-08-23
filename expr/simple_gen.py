#!/usr/bin/env python3
import sys
import os

path = sys.argv[1]

prefix = os.path.dirname(path)
filename = os.path.basename(path)
base = filename
ext = ''
percentage = 0.01

if not os.path.exists('%s/%.2f' % (prefix, percentage)):
    os.mkdir('%s/%.2f' % (prefix, percentage))

fp_vm = open('%s/%s_%.2f.v' % (prefix, base, percentage), 'w')
fp_base = open('%s/%.2f/%s.base' % (prefix, percentage, base), 'w')
fp_append = open('%s/%.2f/%s.update' % (prefix, percentage, base), 'w')
fp_updated = open('%s/%.2f/%s.updated' % (prefix, percentage, base), 'w')

vm_set = set()

with open(path, 'r') as fi:
    line_no = 0
    for line in fi:
        line_no += 1
        parts = line.split()
        vm_set.add(int(parts[0]))
        vm_set.add(int(parts[1]))

    n_line = line_no
    add_size = int(n_line * percentage / 2)
    del_size = int(n_line * percentage / 2)

    base_begin = 0
    base_end = n_line - add_size - del_size
    add_begin = base_end
    add_end = base_end + add_size
    del_begin = add_end
    del_end = del_begin + del_size

    print("Vertex num: ", len(vm_set))
    print("Add size: ", add_size)
    print("Del size: ", del_size)

with open(path, 'r') as fi:
    line_no = 0

    for line in fi:
        if base_begin <= line_no < base_end:
            fp_base.write(line)
            fp_updated.write(line)
        elif add_begin <= line_no < add_end:
            fp_append.write("a " + line)
            fp_updated.write(line)
        elif del_begin <= line_no < del_end:
            fp_base.write(line)
            fp_append.write("d " + line)
        line_no += 1

print("Writing vm")
for v in vm_set:
    fp_vm.write('%d\n' % v)

fp_vm.close()
fp_base.close()
fp_append.close()
fp_updated.close()