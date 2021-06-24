"""
inference_sampler和neighbor_sampler产生的邻居结果的对比图
"""
import sys
import time
import os
import numpy as np
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.data import NeighborSampler
from pyg_utils import get_dataset, get_split_by_file, small_datasets, datasets_maps
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 15
plt.rcParams["font.size"] = base_size
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "coauthor-physics", "com-amazon"]
xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL\nGRAPH']
cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
}


def pics_inference_graph_info(file_name, ylabel, file_class, ymax, dir_save="exp3_thesis_figs/sampling"):
    print(file_name)
    df = pd.read_csv("paper_exp4_relative_sampling/" + file_name, index_col=0)
    
    markers = 'oD^sdp'
    colors = plt.get_cmap('Dark2')(np.linspace(0.15, 0.85, len(small_datasets)))
    
    # inference sampler
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_ylabel(ylabel, fontsize=base_size+2)
    ax.set_xlabel("相对批规模", fontsize=base_size+2)
    # ax.set_ylim(0, ymax)
    plt.xticks(fontsize=base_size+2)
    plt.yticks(fontsize=base_size+2)
    for i, data in enumerate(small_datasets):
        ax.plot(xticklabels, df[f'{data}_inference_sampler'], 
                color=colors[i], marker=markers[i], label=datasets_maps[data])

    ax.set_xticklabels(xticklabels, fontsize=base_size)
    ax.legend(fontsize='small')
    fig.savefig(dir_save + "/exp_inference_sampling_graph_info_inference_sampler_" + file_class + ".png", dpi=400)


pics_inference_graph_info("inference_graph_info_edges.csv", "边数", "edges", 980000)
pics_inference_graph_info("inference_graph_info_avg_degrees.csv", "平均度数", "avg_degrees", 37)