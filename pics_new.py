import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables
plt.style.use("ggplot")


def run_epochs():
    plt.rcParams["font.size"] = 12
    datasets = []
    dir_out = ""
    for data in datasets:
        fig, ax = plt.subplots()
        ax.set_ylabel("Training Time / Epoch (ms)")
        ax.set_xlabel("Algorithm")
        df = pd.read_csv(dir_out + "/epochs/" + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax)
        plt.tight_layout()
        fig.savefig(dir_out + "/exp_absolute_training_time_comparison_" + data + ".png")


# 1. stages, layers, operators, edge-cal
def pic_calculations(file_prefix, dir_out, algs, xticklabels, xlabel, labels):
    plt.rcParams["font.size"] = 12
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    # xlabel = r"#$d_a$ and #$d_v$ (HEAD=4)"
    log_y = True

    dir_path = dir_out + '/calculations'
    for alg in algs:
        plt.figure(figsize=(12, 9.6))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig = plt.figure(1)
        df = {}
        df[0] = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_path + '/' + alg + '_' + data + '.csv' # 这里只与alg和data相关
            if not os.path.exists(file_path):
                continue
            df_t = pd.read_csv(file_path, index_col=0).T
            if df_t.empty:
                df[0][data] = [np.nan] * len(xticklabels)
                df[1][data] = [np.nan] * len(xticklabels)
            else:
                df[0][data] = df_t[0].values.tolist() + [np.nan] * (len(xticklabels) - len(df_t[0].values))
                df[1][data] = df_t[1].values.tolist() + [np.nan] * (len(xticklabels) - len(df_t[1].values))

        df[0] = pd.DataFrame(df[0])
        df[1] = pd.DataFrame(df[1])
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        for i, ax in enumerate([ax1, ax2]):
            ax.set_title(algorithms[alg] + ' ' + labels[i])
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel('Training Time / Epoch (ms)')
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(xticklabels))))
            ax.set_xticklabels(xticklabels)
            markers = 'oD^sdp'
            for j, c in enumerate(df[i].columns):
                df[i][c].plot(ax=ax, marker=markers[j], label=c, rot=0)
            ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_path + "/" + file_prefix + alg + ".png")
        plt.close()


def run_stages():
    dicts = {
        'stages': ['Forward', 'Backward', 'Eval'],
        'layers': ['Layer0', 'Layer1', 'Loss', 'Other'],
        'calculations': ['Vertex-cal', 'Edge-cal'],
        'edge-cal': ['Collect', 'Message', 'Aggregate', 'Update']
    }
    # for GCN and GGNN
    # file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_"
    # xticklabels = ['16', '32', '64', '128', '256', '512', '1024', '2048']
    # xlabel = "Hidden Dimensions"
    # algs = ['gcn', 'ggnn']
    # dir_out = "hidden_dims_exp"
    
    # for GAT head_dims
    # file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_1_"
    # xticklabels = ['8', '16', '32', '64', '128', '256']
    # xlabel = r"#$d_{head}$ (Head=4)"
    # algs = ['gat']
    # dir_out = "hds_head_dims_exp"

    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_2_"    
    xticklabels = ['1', '2', '4', '8', '16']
    xlabel = r"#Head ($d_{head}$=4)"
    algs = ['gat']
    dir_out = "/data/wangzhaokang/wangyunpan/hds_heads_exp"

    # for GaAN head_dims
    # file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_1_"
    # xticklabels = ['8', '16', '32', '64', '128', '256']
    # xlabel = r"#$d_{v}$ and #$d_{a}$(Head=4)"
    # algs = ['gaan']
    # dir_out = "/data/wangzhaokang/wangyunpan/hds_head_dims_exp"

    # file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_2_"    
    # xticklabels = ['1', '2', '4', '8', '16']
    # xlabel = r"#Head ($d_{v}$=4, $d_{a}$=4)"
    # algs = ['gaan']
    # dir_out = "/data/wangzhaokang/wangyunpan/hds_heads_exp"
    
    for label in ['calculations']:
        pic_calculations(file_prefix=file_prefix, dir_out=dir_out, algs=algs,
                   xticklabels=xticklabels, xlabel=xlabel, labels=dicts[label])

run_stages()