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


def run_memory_fix_edge(file_type="png"):
    file_out="exp_memory_expansion_ratio_input_graph_number_of_vertices_fixed_edge_"
    log_y = False    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['graph']
    
    xticklabels = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
    variables = ['1k', '5k', '10k', '20k', '30k', '40k', '50k']
    
    base_path = "exp3_thesis_figs/memory"
    xlabel = "点数"

    file_prefix, file_suffix = '_', '_500k'

    df_peak = pd.read_csv('paper_exp3_memory/' + file_out + "peak_memory.csv", index_col=0)
    df_ratio = pd.read_csv('paper_exp3_memory/' + file_out + "expansion_ratio.csv", index_col=0)
    # peak memory
    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_ylabel("峰值内存 (GB)", fontsize=base_size+2)
    ax.set_xlabel(xlabel, fontsize=base_size+2)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(variables)

    markers = 'oD^sdp'
    for i, c in enumerate(df_peak.columns):
        ax.plot(xticklabels, df_peak[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=base_size)
    fig.savefig(base_path + "/" + file_out + "peak_memory." + file_type)
    
    # expansion ratio
    fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
    ax.set_ylabel("膨胀比例", fontsize=base_size+2)
    ax.set_xlabel(xlabel, fontsize=base_size+2)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)

    markers = 'oD^sdp'
    for i, c in enumerate(df_ratio.columns):
        ax.plot(xticklabels, df_ratio[c], marker=markers[i], label=algorithms[c])
    ax.set_xticks(xticklabels)
    ax.set_xticklabels(variables)
    ax.legend(fontsize=base_size)
    fig.savefig(base_path + "/" + file_out + "expansion_ratio." + file_type, dpi=400)


run_memory_fix_edge()
