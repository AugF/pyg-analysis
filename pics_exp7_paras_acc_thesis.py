import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import autolabel, datasets_maps, algorithms
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed", "amazon-computers", "coauthor-physics", "flickr", "com-amazon"]
datasets_map = ['amp', 'pub', 'amc', 'cph', 'fli', 'cam']

models = ["gcn", "ggnn", "gat", "gaan"]
gcn_ggnn_hds = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"]
gat_hds = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
gat_heads = ["1", "2", "4", "8", "16"]
gaan_hds = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"]
gaan_ds = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
gaan_heads = ["1", "2", "4", "8", "16"]


def pics_gcn_ggnn(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048']
    xlabel = "隐藏向量维度" + r"$dim(\mathbf{h}^1_x)$ "
    algs = ['gcn', 'ggnn']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']

    for alg in algs:
        df = pd.read_csv(dir_in + "/" + alg + "_hds.csv", index_col=0)
        df.index = xticklabels
        fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
        ax.set_ylabel('测试集精度', fontsize=base_size+2)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, fontsize=base_size+2)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            df[c].plot(ax=ax, marker=markers[j], markersize=7, label=c, rot=0)
        ax.legend(ncol=2, fontsize='small')
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=base_size-2)
        fig.savefig(dir_out + "/" + file_prefix + alg + ".png", dpi=400)
        plt.close()


def pics_gat(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
    base_size = 14
    plt.rcParams["font.size"] = base_size
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = [['1', '2', '4', '8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabels =  [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"]
    
    for i, mode in enumerate(['hds', 'heads']):
        df = pd.read_csv(dir_in + "/gat_" + mode + ".csv", index_col=0)
        df.index = xticklabels[i]
        fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
        ax.set_ylabel('测试集精度', fontsize=base_size+2)
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabels[i], fontsize=base_size + 2)
        ax.set_xticks(list(range(len(xticklabels[i]))))
        ax.set_xticklabels(xticklabels[i])
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            df[c].plot(ax=ax, marker=markers[j], markersize=7, label=c, rot=0)
        ax.legend(ncol=2, fontsize='small')
        fig.savefig(dir_out + "/" + file_prefix + "gat_" + mode + ".png", dpi=400)
        plt.close()


def pics_gaan(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
    
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = [['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048'],
                   ['1', '2', '4', '8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabels = ["隐藏向量维度" + r"$dim(\mathbf{h}^1_x)$" + "\n" + r"(#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$" + "\n" + r"(#Head=4, $dim(\mathbf{h}^1_x)$=64)",  "#Head\n" + r"($dim(\mathbf{h}^1)$=64, $d_a=d_v=d_m$=32)"]
  
    for i, mode in enumerate(['hds', 'ds', 'heads']):
        df = pd.read_csv(dir_in + "/gaan_" + mode + ".csv", index_col=0)
        df.index = xticklabels[i]
        fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
        ax.set_ylabel('测试集精度', fontsize=base_size + 2)
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabels[i], fontsize=base_size + 2)
        ax.set_xticks(list(range(len(xticklabels[i]))))
        ax.set_xticklabels(xticklabels[i], rotation=45)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            ax.plot(df.index, df[c], marker=markers[j], markersize=7, label=c)
        ax.legend(loc="center right", ncol=2, fontsize='small')
        fig.savefig(dir_out + "/" + file_prefix + "gaan_" + mode + ".png", dpi=400)
    plt.close()


def pics_max_acc(dir_in, dir_out):
    base_size = 14
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
    
    df_data = pd.read_csv(dir_in + f'/alg_acc.csv', index_col=0)
    df = {}
    for i, alg in enumerate(algs):
        df[i] = [float(x) for x in list(df_data[alg])]

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
    ax.set_ylabel("测试集精度", fontsize=base_size + 2)
    ax.set_xlabel("数据集", fontsize=base_size + 2)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(fontsize=base_size)
    plt.yticks(fontsize=base_size)
    
    for r in rects:
        ax = autolabel(r, ax, memory_ratio_flag=False)
    ax.legend(loc="upper right", ncol=1, fontsize='small')
    fig.savefig(dir_out + "/exp_hyperparameter_on_accuracy_alg_contrast.png", dpi=400)
    plt.close()


dir_in = 'paper_exp1_super_parameters'
dir_out = 'exp3_thesis_figs/paras'
pics_gcn_ggnn(dir_in=dir_in + "/acc_res", dir_out=dir_out)
pics_gat(dir_in=dir_in + "/acc_res", dir_out=dir_out)
pics_gaan(dir_in=dir_in + "/acc_res", dir_out=dir_out)
pics_max_acc(dir_in=dir_in + "/acc_res", dir_out=dir_out)