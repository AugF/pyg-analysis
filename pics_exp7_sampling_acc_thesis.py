import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed",
            "amazon-computers", "coauthor-physics", "flickr"]
modes = ["cluster", "graphsage"]
algs = ["gcn", "ggnn", "gat", "gaan"]

cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625]
}


datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
}

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

dir_in = "paper_exp4_relative_sampling/acc"
dir_out = "exp3_thesis_figs/sampling"


def pics_relative_batch_acc():
    df_full = pd.read_csv("/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp5_paras_acc/acc_res/alg_acc.csv", index_col=0)
    for mode in modes:
        for data in ['amazon-photo', 'amazon-computers', 'flickr']:
            df = pd.read_csv(dir_in + "/" + mode + "_" + data + ".csv", index_col=0)

            markers = "oD^sdp"

            fig, ax = plt.subplots(figsize=(7/3, 6/3), tight_layout=True)
            for i, c in enumerate(df.columns):
                ax.plot(xticklabels, list(df[c]) + [df_full.loc[datasets_maps[data], c]], markersize=4, marker=markers[i], label=algorithms[c])

            ax.set_xlabel("相对批规模", fontsize=10)
            ax.set_ylabel("测试集精度", fontsize=10)
            ax.set_ylim(0.4, 1)
            ax.set_xticklabels(xticklabels, fontsize=8, rotation=30)
            
            ax.legend(fontsize="x-small", ncol=2)
            #ax.legend(loc="upper right", ncol=4, fontsize="medium")
            # ax.legend()
            fig.savefig(dir_out + f"/exp_{mode}_sampling_accuracy_on_{datasets_maps[data]}.png", dpi=400)



pics_relative_batch_acc()
