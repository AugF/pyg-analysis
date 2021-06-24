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


def pics_minibatch_time(file_type="png"):
    base_size = 12
    plt.rcParams["font.size"] = base_size
    dir_out = "exp3_thesis_figs/sampling"
    file_out = "exp_sampling_relative_batch_size_train_time_stack_"
    
    ylabel = "每批次训练时间 (ms)"
    xlabel = "相对批规模"

    cluster_batchs = [15, 45, 90, 150, 375, 750]

    graphsage_batchs = {
        'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
        'flickr': [893, 2678, 5355, 8925, 22313, 44625],
    }
    
    xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    
    for mode in ['cluster', 'graphsage']:
        for data in ["amazon-computers", "flickr"]:
            df_train = pd.read_csv("paper_exp4_relative_sampling/batch_time_stack/" + file_out + mode + "_" + data + "_train_time.csv", index_col=0)
            df_to = pd.read_csv("paper_exp4_relative_sampling/batch_time_stack/" + file_out + mode + "_" + data + "_to_time.csv", index_col=0)
            df_sampling = pd.read_csv("paper_exp4_relative_sampling/batch_time_stack/" + file_out + mode + "_" + data + "_sampling_time.csv", index_col=0)
                    
            fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
            ax.set_ylabel(ylabel, fontsize=base_size+2)
            ax.set_xlabel(xlabel, fontsize=base_size+2)

            enabels_indexs = [] 
            labels = []           
            for i, var in enumerate(xticklabels):
                flag = False
                for alg in algs:
                    if str(df_train[alg][i]) != 'nan' or str(df_to[alg][i]) != 'nan' or str(df_sampling[alg][i]) != 'nan':
                        flag = True
                        break
                if flag:
                    enabels_indexs.append(i)
                    labels.append(var)
            
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=base_size+2)
            file_name = mode + '_' + data

            locations = [-1.5, -0.5, 0.5, 1.5]
            x = np.arange(len(labels))
            width = 0.2
            colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))

            rects = []
            i = 0
            for alg in algs:
                tmp_train = [df_train[alg][j] for j in enabels_indexs]
                tmp_to = [df_to[alg][j] for j in enabels_indexs]
                tmp_sampling = [df_sampling[alg][j] for j in enabels_indexs]
                rects.append(ax.bar(x + locations[i] * width, tmp_train, width, color=colors[i], edgecolor='black', hatch="////"))
                
                rects.append(ax.bar(x + locations[i] * width, tmp_to, width, color=colors[i], edgecolor='black', bottom=tmp_train, hatch='....'))
                tmp = [tmp_train[j] + tmp_to[j] for j in range(len(tmp_train))]
                rects.append(ax.bar(x + locations[i] * width, tmp_sampling, width, color=colors[i], edgecolor='black', bottom=tmp, hatch='xxxx'))
                i += 1
            
            legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
            legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='////'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='xxxx')]
            ax.legend(legend_colors + legend_hatchs, [algorithms[i] for i in algs] + ['GPU训练', '数据传输', '采样'], fontsize=base_size, ncol=2)
            fig.savefig(dir_out + '/' + file_out + mode + "_" + data + "." + file_type, dpi=400)
            plt.close()


def pics_inference_sampling_minibatch_time(dir_work="inference_sampling_time", file_suffix="", file_type="png"):
    base_size = 12
    plt.rcParams["font.size"] = base_size
    dir_out = "exp3_thesis_figs/sampling"
    file_out = "exp_inference_sampling_fix_batch_size_train_time_stack_" + file_suffix
    
    ylabel = "每批次推理时间 (ms)"
    xlabel = "数据集"
    
    xticklabels = [datasets_maps[i] for i in datasets]

    algs = ['gcn', 'ggnn', 'gat', 'gaan']

    df_train = pd.read_csv("paper_exp4_relative_sampling/batch_time_stack/" + file_out + "_train_time.csv", index_col=0)
    df_to = pd.read_csv("paper_exp4_relative_sampling/batch_time_stack/" + file_out + "_to_time.csv", index_col=0)
    df_sampling = pd.read_csv("paper_exp4_relative_sampling/batch_time_stack/" + file_out + "_sampling_time.csv", index_col=0)
   
    fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    ax.set_ylabel(ylabel, fontsize=base_size+2)
    ax.set_xlabel(xlabel, fontsize=base_size+2)
    ax.set_ylim(0, 120)
    ax.set_xticklabels([''] + xticklabels, fontsize=base_size+2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(datasets))
    width = 0.2
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))

    rects = []
    i = 0
    for alg in algs:
        tmp_train, tmp_to, tmp_sampling = df_train[alg], df_to[alg], df_sampling[alg]
        rects.append(ax.bar(x + locations[i] * width, tmp_train, width, color=colors[i], edgecolor='black', hatch="////"))
        rects.append(ax.bar(x + locations[i] * width, tmp_to, width, color=colors[i], edgecolor='black', bottom=tmp_train, hatch='....'))
        tmp = [tmp_train[j] + tmp_to[j] for j in range(len(tmp_train))]
        rects.append(ax.bar(x + locations[i] * width, tmp_sampling, width, color=colors[i], edgecolor='black', bottom=tmp, hatch='xxxx'))
        i += 1

    legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
    legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='////'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='xxxx')]
    ax.legend(legend_colors + legend_hatchs, [algorithms[i] for i in algs] + ['GPU推理', '数据传输', '采样'], ncol=2, loc="upper left", fontsize=base_size)
    
    fig.savefig(dir_out + '/' + file_out +  "." + file_type, dpi=400)
    plt.close()


pics_minibatch_time(file_type="png")
pics_inference_sampling_minibatch_time(dir_work="inference_sampling_time_2048", file_suffix="2048", file_type="png")
pics_inference_sampling_minibatch_time(dir_work="inference_sampling_time", file_suffix="1024", file_type="png")
