import sys

input_mtx = sys.argv[1]

if not input_mtx.endswith(".mtx"):
    sys.exit("It's not a .mtx")

print("Counting...")
n_self_cycle = 0

with open(input_mtx, 'r') as fi:
    for line in fi:
        line = line.strip()
        if line.startswith("%"):
            continue
        arr = line.split()
        if len(arr) > 1:
            if arr[0] == arr[1]:
                n_self_cycle += 1

print("n self cycle: %s" % n_self_cycle)
