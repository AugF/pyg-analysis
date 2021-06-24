import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, autolabel, datasets_maps
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

def run_memory_factors_dense_feats_dims(file_type="png"):
    file_out="exp_memory_expansion_ratio_input_feature_dimension_"
    log_y = True    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['com-amazon']
    variables = [16, 32, 64, 128, 256, 512]

    dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp3_memory/dir_feat_dims_json"
    base_path = "exp3_thesis_figs/memory"
    
    file_prefix, file_suffix = '_', ''
    
    for data in datasets:
        df_ratio = pd.read_csv('paper_exp3_memory/' + file_out + data + '.csv', index_col=0)
        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        ax.set_ylabel("膨胀比例", fontsize=base_size + 2)
        ax.set_xlabel("输入特征维度", fontsize=base_size + 2)
        plt.xticks(fontsize=base_size)
        plt.yticks(fontsize=base_size)
        ax.set_xticks(np.arange(len(variables)))
        ax.set_xticklabels(variables)
        df_ratio = pd.DataFrame(df_ratio)
        if log_y:
            ax.set_yscale("symlog", basey=2)
         
        locations = [-1.5, -0.5, 0.5, 1.5]
        x = np.arange(len(variables))
        width = 0.2
        rects = []
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
        
        i = 0
        for (col, c) in zip(df_ratio.columns, colors):
            rects.append(ax.bar(x + locations[i] * width, df_ratio[col], width, label=algorithms[algs[i]], color=c, edgecolor='black'))
            i += 1
        ax.legend(fontsize=12)
        fig.savefig(base_path + "/" + file_out + data + "." + file_type, dpi=400)


run_memory_factors_dense_feats_dims(file_type="png")
