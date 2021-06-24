# coding=utf-8
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algorithms, datasets_maps
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

def pics_memory(file_type="png"):
    dir_out = "exp3_thesis_figs/memory"
    # time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Loss', 'Backward',
    #                'Eval\nLayer0', 'Eval\nLayer1']
    time_labels = ['数据加载', 'GPU预热', '前向传播\nLayer0', '前向传播\nLayer1', '损失计算', '后向传播', '评估\nLayer0', '评估\nLayer1']

    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo']
    for i, data in enumerate(datasets):
        allocated_max = pd.read_csv('paper_exp3_memory/exp_memory_usage_stage_' + datasets_maps[data] + '.csv', index_col=0)
        #ax = plt.subplot(2, 3, i + 1)
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_ylabel("峰值内存 (MB)", fontsize=base_size+2)
        ax.set_xlabel("时期", fontsize=base_size+2)
        plt.xticks(fontsize=base_size)
        plt.yticks(fontsize=base_size)
        ax.set_ylim(0, 2500)
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        x = np.arange(len(time_labels))
        for j, c in enumerate(allocated_max.columns):
            ax.plot(x, allocated_max[c], marker=markers[j], linestyle=lines[j], label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(time_labels, fontsize=base_size-1, rotation=45)
        ax.legend(fontsize=base_size)
        fig.tight_layout() 
        fig.savefig(dir_out + '/exp_memory_usage_stage_' + datasets_maps[data] + '.' + file_type, dpi=400)
        plt.close()


pics_memory(file_type="png")
