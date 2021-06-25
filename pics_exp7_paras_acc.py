import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import autolabel, datasets_maps, algorithms
from matplotlib.font_manager import _rebuild
_rebuild() 

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.rcParams["font.size"] = 12
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

def save_acc_to_csv(dir_work="early_stopping2"): # 将统计得到的acc结果保存在csv文件中 
    if not os.path.exists(f"{dir_work}/acc_res"):
        os.makedirs(f"{dir_work}/acc_res")
    for model in models:
        if model == "gcn" or model == "ggnn":
            dir_config = "dir_gcn_ggnn_acc"
            # hds
            df_hds = {}
            for data in datasets:
                df_hds[data] = []
                for hd in gcn_ggnn_hds:
                    file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_{hd}.log"
                    if not os.path.exists(file_name):
                        df_hds[data].append(None)
                        continue
                    print(file_name)
                    acc = None                    
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test Acc: (.*)", line)
                            if match_line:
                                acc = format(float(match_line.group(1)), ".5f")
                                break
                    df_hds[data].append(acc)
            df = pd.DataFrame(df_hds, index=gcn_ggnn_hds)
            df.columns = datasets_map
            df.to_csv(f"{dir_work}/acc_res/{model}_hds.csv")
        elif model == "gat":
            dir_config = "dir_gat_acc"
            # hds
            df_hds = {}
            for data in datasets:
                df_hds[data] = []
                for hd in gat_hds:
                    file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_4_{hd}.log"
                    if not os.path.exists(file_name):
                        df_hds[data].append(None)
                        continue   
                    print(file_name)
                    acc = None
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test Acc: (.*)", line)
                            if match_line:
                                acc = format(float(match_line.group(1)), ".5f")
                                break
                    df_hds[data].append(acc)
            df = pd.DataFrame(df_hds, index=gat_hds)
            df.columns = datasets_map
            df.to_csv(f"{dir_work}/acc_res/{model}_hds.csv")
            
            # heads
            df_heads = {}
            for data in datasets:
                df_heads[data] = []
                for h in gat_heads:
                    file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_{h}_32.log"
                    if not os.path.exists(file_name):
                        df_heads[data].append(None)
                        continue   
                    print(file_name)
                    acc = None
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test Acc: (.*)", line)
                            if match_line:
                                acc = format(float(match_line.group(1)), ".5f")
                                break
                    df_heads[data].append(acc)
            df = pd.DataFrame(df_heads, index=gat_heads)
            df.columns = datasets_map
            df.to_csv(f"{dir_work}/acc_res/{model}_heads.csv")
        elif model == "gaan":
            dir_config = "dir_gaan_acc"
            # hds
            print("gaan hds")
            df_hds = {}
            for data in datasets:
                df_hds[data] = []
                for hd in gaan_hds:
                    file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_4_32_{hd}.log"
                    if not os.path.exists(file_name):
                        df_hds[data].append(None)
                        continue   
                    print(file_name)
                    acc = None
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test Acc: (.*)", line)
                            if match_line:
                                acc = format(float(match_line.group(1)), ".5f")
                                break
                    df_hds[data].append(acc)
            df = pd.DataFrame(df_hds, index=gaan_hds)
            df.columns = datasets_map
            df.to_csv(f"{dir_work}/acc_res/{model}_hds.csv")
            
            # ds
            print("gaan ds")
            df_ds = {}
            for data in datasets:
                print("ds", model, data, gaan_ds)
                df_ds[data] = []
                for d in gaan_ds:
                    file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_4_{d}_64.log"
                    if not os.path.exists(file_name):
                        df_ds[data].append(None)
                        continue
                    print(file_name)
                    acc = None
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test Acc: (.*)", line)
                            if match_line:
                                acc = format(float(match_line.group(1)), ".5f")
                                break
                    df_ds[data].append(acc)
                
            df = pd.DataFrame(df_ds, index=gaan_ds)
            df.columns = datasets_map
            df.to_csv(f"{dir_work}/acc_res/{model}_ds.csv")
                
            # heads
            print("gaan heads")
            df_heads = {}
            for data in datasets:
                df_heads[data] = []
                for h in gaan_heads:
                    file_name = f"{dir_work}/{dir_config}/config0_{model}_{data}_{h}_32_64.log"
                    if not os.path.exists(file_name):
                        df_heads[data].append(None)
                        break
                    print(file_name)
                    acc = None
                    with open(file_name) as f:
                        for line in f:
                            match_line = re.match("Final Test Acc: (.*)", line)
                            if match_line:
                                acc = format(float(match_line.group(1)), ".5f")
                                break
                    df_heads[data].append(acc)
            df = pd.DataFrame(df_heads, index=gaan_heads)
            df.columns = datasets_map
            df.to_csv(f"{dir_work}/acc_res/{model}_heads.csv")


def pics_gcn_ggnn(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048']
    xlabel = "隐藏向量的维度"
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
            df[c].plot(ax=ax, marker=markers[j], label=c, rot=0)
        ax.legend(ncol=2, fontsize='small')
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=base_size-2)
        fig.savefig(dir_out + "/" + file_prefix + alg + ".png")
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
            df[c].plot(ax=ax, marker=markers[j], markersize=8, label=c, rot=0)
        ax.legend(ncol=2, fontsize='small')
        fig.savefig(dir_out + "/" + file_prefix + "gat_" + mode + ".png")
        plt.close()


def pics_gaan(dir_in="early_stopping1/acc_res", dir_out="early_stopping1/acc_res"):
    base_size = 9
    plt.rcParams["font.size"] = base_size
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr']
    
    file_prefix = "exp_hyperparameter_on_accuracy_"
    xticklabels = [['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048'],
                   ['1', '2', '4', '8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabels = ["隐藏向量的维度\n" + r"$dim(\mathbf{h}^1_x)$ (#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$" + "\n" + r"(#Head=4, $dim(\mathbf{h}^1_x)$=64)",  "#Head\n" + r"($dim(\mathbf{h}^1)$=64, $d_a=d_v=d_m$=32)"]
  
    fig, axes = plt.subplots(1, 3, figsize=(7, 8/3), sharey=True, tight_layout=True) 
    # plt.rcParams["font.size"] = 8
    for i, mode in enumerate(['hds', 'ds', 'heads']):
        df = pd.read_csv(dir_in + "/gaan_" + mode + ".csv", index_col=0)
        df.index = xticklabels[i]
        ax = axes[i]
        if i == 0:
            ax.set_ylabel('测试集精度', fontsize=base_size + 2)
        ax.set_ylim(0.4, 1)
        ax.set_xlabel(xlabels[i], fontsize=base_size + 2)
        ax.set_xticks(list(range(len(xticklabels[i]))))
        ax.set_xticklabels(xticklabels[i], rotation=45)
        markers = 'oD^sdp'
        for j, c in enumerate(df.columns[:-1]):
            ax.plot(df.index, df[c], marker=markers[j], markersize=7, label=c)
        ax.legend(loc="center right", ncol=2, fontsize='small')
    fig.savefig(dir_out + "/" + file_prefix + "gaan.png")
    # fig.savefig(dir_out + "/" + file_prefix + "gaan.pdf")
    plt.close()


# 纵向结果对比
def get_alg_contrast(dir_in="early_stopping2/acc_res", dir_out="early_stopping2/acc_res"):
    acc_data = {}
    acc_maxv = {}
    for alg in models:
        if alg in ["gcn", "ggnn"]:
            df = pd.read_csv(dir_in + "/" + alg + "_hds.csv", index_col=0)
            acc_data[alg] = np.max(df.fillna(0).values, axis=0)[:-1]
            acc_maxv[alg] = [df.index[i] for i in np.argmax(df.fillna(0).values, axis=0)[:-1]]
        elif alg == "gat":
            df = pd.read_csv(dir_in + "/" + alg + "_hds.csv", index_col=0)
            acc_data[alg] = np.max(df.fillna(0).values, axis=0)[:-1]
            acc_maxv[alg] = ["4_" + str(df.index[i]) for i in np.argmax(df.fillna(0).values, axis=0)[:-1]]
            
            df = pd.read_csv(dir_in + "/" + alg + "_heads.csv", index_col=0)
            for i, j in enumerate(np.argmax(df.fillna(0).values, axis=0)[:-1]):
                if acc_data[alg][i] < df.values[j][i]:
                    acc_data[alg][i] = df.values[j][i]
                    acc_maxv[alg][i] = str(df.index[j]) + "_32"
        elif alg == "gaan":
            df = pd.read_csv(dir_in + "/" + alg + "_hds.csv", index_col=0)
            acc_data[alg] = np.max(df.fillna(0).values, axis=0)[:-1]
            acc_maxv[alg] = ["4_32_" + str(df.index[i]) for i in np.argmax(df.fillna(0).values, axis=0)[:-1]]
            
            df = pd.read_csv(dir_in + "/" + alg + "_heads.csv", index_col=0)
            for i, j in enumerate(np.argmax(df.fillna(0).values, axis=0)[:-1]):
                if acc_data[alg][i] < df.values[j][i]:
                    acc_data[alg][i] = df.values[j][i]
                    acc_maxv[alg][i] = str(df.index[j]) + "_32_64"       
            
            df = pd.read_csv(dir_in + "/" + alg + "_ds.csv", index_col=0)
            for i, j in enumerate(np.argmax(df.fillna(0).values, axis=0)[:-1]):
                if acc_data[alg][i] < df.values[j][i]:
                    acc_data[alg][i] = df.values[j][i]
                    acc_maxv[alg][i] = "4_" + str(df.index[j]) + "_64"     
    pd.DataFrame(acc_data, index=datasets_map[:-1]).to_csv(dir_out + "/alg_acc.csv")
    pd.DataFrame(acc_maxv, index=datasets_map[:-1]).to_csv(dir_out + "/max_acc.csv")
    print(acc_maxv)


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
        rects.append(ax.bar(x + l * width, df[i], width, label=algorithms[algs[i]], color=c))
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
    ax.legend(loc="upper right", ncol=1)
    fig.savefig(dir_out + "/exp_hyperparameter_on_accuracy_alg_contrast.png")
    plt.close()


# save_acc_to_csv("./")
dir_in = '/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp5_paras_acc'
dir_out = '/mnt/data/wangzhaokang/wangyunpan/pyg-analysis/paper_exp1_super_parameters'
pics_gcn_ggnn(dir_in=dir_in + "/acc_res", dir_out=dir_out + "/acc_figs")
pics_gat(dir_in=dir_in + "/acc_res", dir_out=dir_out + "/acc_figs")
pics_gaan(dir_in=dir_in + "/acc_res", dir_out=dir_out + "/acc_figs")
pics_max_acc(dir_in=dir_in + "/acc_res", dir_out=dir_out + "/acc_figs")