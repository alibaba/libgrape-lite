import sys

input_mtx = sys.argv[1]
input_v = sys.argv[2]
output_v = sys.argv[3]

if not input_mtx.endswith(".mtx"):
    sys.exit("It's not a .mtx")

line_no = 0
in_dgr = {}
out_dgr = {}

print("Counting degree")

with open(input_mtx, 'r') as fi:
    for line in fi:
        line = line.strip()
        if line.startswith("%"):
            continue
        line_no += 1
        if line_no == 1:
            continue
        arr = line.split()
        if len(arr) > 1:
            src = int(arr[0])
            dst = int(arr[1])
            in_dgr[dst] = in_dgr.get(dst, 0) + 1
            out_dgr[src] = out_dgr.get(src, 0) + 1

print("Attaching degree")
with open(input_v, 'r') as fi:
    with open(output_v, 'w') as fo:
        for line in fi:
            line = line.strip()
            arr = line.split()
            if len(arr) > 0:
                v = int(arr[0])
                fo.write("%s %d %d\n" % (line, in_dgr.get(v, 0), out_dgr.get(v, 0)))
