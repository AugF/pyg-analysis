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

def run_memory_degrees(file_type="png", dir_save="exp3_thesis_figs/memory"):
    file_out="exp_memory_expansion_ratio_input_graph_number_of_edges_"
    log_y = False    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['graph']
    
    variables = [2, 5, 10, 15, 20, 30, 40, 50, 70]
    xticklabels = variables
    
    dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp3_memory/dir_degrees_json"
    base_path = "paper_exp3_memory"
    xlabel = "平均度数"

    file_prefix, file_suffix = '_10k_', ''
    
    df_peak = pd.read_csv('paper_exp3_memory/' + file_out + "peak_memory.csv", index_col=0)
    df_ratio = pd.read_csv('paper_exp3_memory/' + file_out + "expansion_ratio.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_ylabel("峰值内存 (GB)", fontsize=base_size+2)
    ax.set_xlabel(xlabel, fontsize=base_size+2)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    # ax.set_xticks(xticklabels)
    # ax.set_xticklabels(variables)
    
    markers = 'oD^sdp'
    for i, c in enumerate(df_peak.columns):
        ax.plot(xticklabels, df_peak[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=12)
    fig.savefig(dir_save + "/" + file_out + "peak_memory." + file_type, dpi=400)
    
    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_ylabel("膨胀比例", fontsize=base_size+2)
    ax.set_xlabel(xlabel, fontsize=base_size+2)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    # ax.set_xticks(xticklabels)
    # ax.set_xticklabels(variables)

    markers = 'oD^sdp'
    for i, c in enumerate(df_ratio.columns):
        ax.plot(xticklabels, df_ratio[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=12)
    fig.savefig(dir_save + "/" + file_out + "expansion_ratio." + file_type, dpi=400)


run_memory_degrees()