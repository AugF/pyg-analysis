import os
import re
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import algorithms
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 12
plt.rcParams["font.size"] = base_size

nodes = ['05k', '1k', '2k', '4k', '6k', '8k', '10k', '15k']
numbers = [500, 1000, 2000, 4000, 6000, 8000, 10000, 15000]
algs = ['gcn', 'ggnn', 'gat', 'gaan']

file_out = "exp_small_graph_train_time"
xlabel = "点数"
ylabel = '每轮训练时间 (ms)'

df_mean = pd.read_csv('paper_exp4_relative_sampling/' + file_out + '_mean.csv', index_col=0)
df_std = pd.read_csv('paper_exp4_relative_sampling/' + file_out + '_std.csv', index_col=0)

fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
ax.set_xlabel(xlabel, fontsize=base_size+2)
ax.set_ylabel(ylabel, fontsize=base_size+2)

colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, len(algs)))
markers = 'oD^sdp'
for i, alg in enumerate(algs):
    ax.errorbar(numbers, df_mean[alg], yerr=df_std[alg], label=algorithms[alg], marker=markers[i], color=colors[i])

ax.legend(fontsize=base_size-2)
fig.savefig("exp3_thesis_figs/sampling/" + file_out + "png", dpi=400)

