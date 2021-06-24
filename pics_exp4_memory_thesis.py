# coding=utf-8
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from utils import algorithms, datasets_maps, datasets, autolabel
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 18
plt.rcParams["font.size"] = base_size


def pics_minibatch_memory_bar(file_type="png"):
    file_out = "exp_sampling_memory_usage_relative_batch_size_"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']

    dir_path = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp4_relative_sampling/batch_memory/"
    dir_out = "exp3_thesis_figs/sampling"
    xlabel = "相对批规模"

    cluster_batchs = [15, 45, 90, 150, 375, 750]

    graphsage_batchs = {
        'amazon-photo': [77, 230, 459, 765, 1913, 3825],
        'pubmed': [198, 592, 1184, 1972, 4930, 9859],
        'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
        'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
        'flickr': [893, 2678, 5355, 8925, 22313, 44625],
        'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
    }

    xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

    log_y = True
    algs = ['gcn', 'ggnn', 'gat', 'gaan']

    for mode in ['cluster', 'graphsage']:
        for data in ["amazon-computers", "flickr"]:
            df_peak = pd.read_csv("paper_exp4_relative_sampling/batch_memory/" + file_out + mode + '_' +
                        data + "_peak_memory.csv", index_col=0)
            # 得到有效index, 去除无效index
            enabels_indexs = []
            labels = []
            for i, var in enumerate(xticklabels):
                flag = False
                for alg in algs:
                    if str(df_peak[alg][i]) != 'nan':
                        flag = True
                        break
                if flag:
                    enabels_indexs.append(i)
                    labels.append(var)

            # 指定bar的location
            locations = [-1.5, -0.5, 0.5, 1.5]
            x = np.arange(len(labels))
            width = 0.2
            colors = plt.get_cmap('Paired')(
                np.linspace(0.15, 0.85, len(locations)))

            fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
            ax.set_ylabel("单个批次峰值内存 (MB)", fontsize=base_size+2)
            ax.set_xlabel(xlabel, fontsize=base_size+2)
            plt.xticks(fontsize=base_size)
            plt.yticks(fontsize=base_size)
            # if log_y:
            #     ax.set_yscale("symlog", basey=2)

            for i, alg in enumerate(algs):
                ax.bar(x + locations[i] * width, [df_peak[alg][d]
                                                  for d in enabels_indexs], width, label=algorithms[alg], color=colors[i], edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=base_size)
            ax.legend(fontsize=base_size)
            fig.savefig(dir_out + "/" + file_out + mode + '_' +
                        data + "_peak_memory." + file_type, dpi=400)
            plt.close()


pics_minibatch_memory_bar(file_type="png")

