import sys

input_mtx = sys.argv[1]

if not input_mtx.endswith(".mtx"):
    sys.exit("It's not a .mtx")

line_no = 0
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
            out_dgr[src] = out_dgr.get(src, 0) + 1

top_v = list(sorted(out_dgr, key=out_dgr.get, reverse=True))[:10]

print("V OD")
for v in top_v:
    print("%d %d" % (v, out_dgr[v]))
