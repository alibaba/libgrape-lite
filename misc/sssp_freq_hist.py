import numpy as np
import matplotlib.pyplot as plt

x = []
limit = 0

with open('/Users/liang/livej_sssp_stat_sortby_freq', 'r') as fi:
    for line in fi:
        x.append(int(line.split()[1]))
        limit += 1

# the histogram of the data
n, bins, patches = plt.hist(x, 13, density=False, stacked=False, facecolor='g', alpha=0.8)
plt.xlabel('Active Times')
plt.ylabel('Count')
plt.title('SSSP Active Freq')
plt.xlim(0, 12)
plt.grid(True)
plt.show()
