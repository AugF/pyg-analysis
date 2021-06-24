import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps
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


def autolabel(rects, ax, memory_ratio_flag=False):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if str(height) == 'nan':
            print(str(height))
            ax.text(rect.get_x() + 0.001, 0.41, "Out Of Memory", fontsize=base_size-6, rotation=90)
            continue
        if memory_ratio_flag:
            ax.text(rect.get_x() , height + 1, f"{height:.1f}", fontsize=base_size-6)
        else:
            ax.text(rect.get_x() + 0.001 , height + 0.02, f"{height:.2f}", fontsize=base_size-6, rotation=90)

    return ax


def run_memory_ratio_config_single(file_type="png"):
    file_out="exp_memory_expansion_ratio"
    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    dir_out = "exp3_thesis_figs/memory"
    
    df = pd.read_csv('paper_exp3_memory/' + file_out + '.csv', index_col=0).values

    labels = [datasets_maps[d] for d in datasets]
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(labels))
    width = 0.2
    rects = []
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    i = 0
    for (l, c) in zip(locations, colors):
        rects.append(ax.bar(x + l * width, df[i], width, label=algorithms[algs[i]], color=c, edgecolor='black'))
        i += 1
    ax.set_ylabel("膨胀比例", fontsize=base_size+2)
    ax.set_xlabel("数据集", fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    
    for r in rects:
        ax = autolabel(r, ax, memory_ratio_flag=True)
    ax.legend(loc="upper right", ncol=2, fontsize=base_size-2)
    fig.savefig(dir_out + "/" + file_out +  "." + file_type, dpi=400)
    plt.close()


def run_inference_full_memory_ratio(file_type="png"):
    file_out="exp_inference_full_memory_expansion_ratio"
    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    dir_out = "exp3_thesis_figs/memory"
    
    df = pd.read_csv('paper_exp3_memory/' + file_out + '.csv', index_col=0).values

    labels = [datasets_maps[d] for d in datasets]
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(labels))
    width = 0.2
    rects = []
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    i = 0
    for (l, c) in zip(locations, colors):
        rects.append(ax.bar(x + l * width, df[i], width, label=algorithms[algs[i]], color=c, edgecolor='black'))
        i += 1
    ax.set_ylabel("膨胀比例", fontsize=base_size+2)
    ax.set_xlabel("数据集", fontsize=base_size+2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 106)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)

    for r in rects:
        ax = autolabel(r, ax, memory_ratio_flag=True)
    ax.legend(loc="upper right", ncol=2, fontsize=base_size-2)
    fig.savefig(dir_out + "/" + file_out +  "." + file_type, dpi=400)
    plt.close()
    

run_memory_ratio_config_single(file_type="png")
run_inference_full_memory_ratio(file_type="png")
