import os
import sys
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import datasets_maps
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 8
plt.rcParams["font.size"] = base_size

small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
}

dir_in = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp4_relative_sampling/batch_graph_info"
dir_out = "exp3_thesis_figs/sampling"
file_out = "exp_sampling_minibatch_realtive_graph_info_"
xticklabels = [1, 3, 6, 10, 25, 50]

for sampler in ["cluster", "graphsage"]:
    df_nodes = pd.read_csv('paper_exp4_relative_sampling/' + file_out + sampler + '_gcn_nodes.csv')
    df_edges = pd.read_csv('paper_exp4_relative_sampling/' + file_out + sampler + '_gcn_edges.csv')
    df_degrees = pd.read_csv('paper_exp4_relative_sampling/' + file_out + sampler + '_gcn_degrees.csv')

    file_names = ['_vertices', '_edges', '_avg_degree']
    ylabels = ["点数", "边数", "平均度数"]
    dfs = [df_nodes, df_edges, df_degrees]
    xlabel = "相对批规模 (%)"

    for k in range(3):
        fig, ax = plt.subplots(figsize=(7/3, 6/3), tight_layout=True)
        df = dfs[k]
        markers = 'oD^sdp'
        colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, len(small_datasets)))
        for i, data in enumerate(small_datasets):
            ax.plot(xticklabels, df[data], 
                    color=colors[i], marker=markers[i], markersize=4, label=datasets_maps[data])
        ax.set_xlabel(xlabel, fontsize=base_size+2)
        ax.set_ylabel(ylabels[k], fontsize=base_size+2)
        ax.legend(fontsize="small")
        fig.savefig(dir_out + "/" + file_out + sampler + file_names[k] + "_gcn.png", dpi=400)        