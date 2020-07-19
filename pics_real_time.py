import os
import re
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import algorithms
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

nodes = ['0.5k', '1k', '2k', '4k', '6k', '8k', '10k', '15k']
numbers = [500, 1000, 2000, 4000, 6000, 8000, 10000, 15000]
algs = ['gcn', 'ggnn', 'gat', 'gaan']
degrees = 4

dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/config_real_time_new_exp'

n = 50
x, y = (n + 1) * 0.25, (n + 1) * 0.75
tx, ty = math.floor(x), math.floor(y) 

df_mean = {}
df_std = {}
for alg in algs:
    df_mean[alg] = []
    df_std[alg] = []
    for ns in nodes:
        file_path = dir_path + '/' + alg + '_graph_' + ns + '_4.log'
        print(file_path)
        with open(file_path) as f:
            tmp = []
            for line in f:
                match_line = re.match(".*, train_time: (.*)s", line)
                if match_line:
                   tmp.append(float(match_line.group(1)))
            tmp.sort()
            Q1 = tmp[tx - 1] * (x - tx) + tmp[tx] * (1 - x + tx)
            Q3 = tmp[ty - 1] * (y - ty) + tmp[ty] * (1 - y + ty)
            min_val, max_val = Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1)
            res = [x for x in tmp if x > min_val and x < max_val]
            mean_time = np.mean(res)
            std_time = np.std(res)
            df_mean[alg].append(mean_time)
            df_std[alg].append(std_time)

file_out = "exp_small_graph_train_time.png"
xlabel = "Number of Vertices"
ylabel = 'Training Time per Epoch (ms)'

fig, ax = plt.subplots()
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, len(algs)))
markers = 'oD^sdp'
for i, alg in enumerate(algs):
    ax.errorbar(numbers, df_mean[alg], yerr=df_std[alg], label=algorithms[alg], marker=markers[i], color=colors[i])

ax.legend()
plt.tight_layout()
fig.savefig(file_out)
