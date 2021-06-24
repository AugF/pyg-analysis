import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib.patches import Polygon, Patch
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 14
plt.rcParams["font.size"] = base_size
small_datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
re_percents = [1024, 2048, 4096, 8192]
dir_path = 'exp3_thesis_figs/sampling'

datasets =  ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'reddit', 'yelp']
color = dict(medians='red', boxes='blue', caps='black')

batch_sizes = [1024, 2048, 4096]
colors = ['black', 'white']

for data in ['amazon-computers', 'flickr', 'yelp']:
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    box_data = []
    for bs in batch_sizes:
        for alg in ['gat', 'gcn']:
            file_path = os.path.join('paper_exp4_relative_sampling', 'batch_inference_memory_safe', '_'.join([alg, data, str(bs)]) + '.csv')
            res = pd.read_csv(file_path, index_col=0)['memory'].values.tolist()
            box_data.append(list(map(lambda x: x/(1024*1024), res)))
    
    if box_data == []:
        continue
    print(box_data)
    bp = ax.boxplot(box_data, patch_artist=True)
    numBoxes = len(batch_sizes) * 2
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        if i % 2 == 1:
            plt.setp(bp['medians'][i], color='red')
            plt.setp(bp['boxes'][i], color='red')
            plt.setp(bp['boxes'][i], facecolor=colors[1])
            plt.setp(bp['fliers'][i], markeredgecolor='red')
        else:
            plt.setp(bp['boxes'][i], facecolor=colors[0])
    
    ax.set_title(data, fontsize=base_size+2)
    ax.set_xlabel('批规模', fontsize=base_size+2)
    ax.set_ylabel('峰值内存 (MB)', fontsize=base_size+2)
    ax.set_xticks([1.5, 3.5, 5.5])
    ax.set_xticklabels(batch_sizes, fontsize=base_size+2)

    legend_colors = [Patch(facecolor='black', edgecolor='black'), Patch(facecolor='white', edgecolor='red')]
    ax.legend(legend_colors, ['GCN', 'GAT'], fontsize=14)
    fig.savefig(f'exp3_thesis_figs/sampling/exp_inference_sampling_memory_fluctuation_{data}.png', dpi=400)
    plt.close()