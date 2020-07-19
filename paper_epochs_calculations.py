import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def run_epochs():
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    dir_out = "config_exp"
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
def pic_calculations_gcn_ggnn():
    # for GCN and GGNN
    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_"
    xticklabels = ['16', '32', '64', '128', '256', '512', '1024', '2048']
    xlabel = "Hidden Dimension"
    algs = ['gcn', 'ggnn']
    labels = ['Vertex Calculation', 'Edge Calculation']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    dir_out = "hidden_dims_exp"
    for alg in algs:
        plt.figure(figsize=(12, 4.8))
        fig = plt.figure(1)
        df = {}
        df[0] = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_out + '/gcn_ggnn_exp/hds_exp/calculations/' + alg + '_' + data + '.csv' # 这里只与alg和data相关
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
            ax.set_title(labels[i])
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel('Training Time / Epoch (ms)')
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(xticklabels))))
            ax.set_xticklabels(xticklabels)
            markers = 'oD^sdp'
            for j, c in enumerate(df[i].columns):
                df[i][c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
            ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + ".pdf")
        plt.close()


def pic_calculations_gaan(file_type="png"):
    labels = ['Vertex Calculation', 'Edge Calculation']
    algs = ['gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_"
    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], ['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabel = [r"Hidden Vector Dimension $dim(\boldsymbol{h}^1)$ (#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$(#Head=4, $dim(\boldsymbol{h}^1)$=64)",  r"#Head ($dim(\boldsymbol{h}^1)$=4, $d_a=d_v=d_m$=32)"]
    dir_out = ["hds_exp", "hds_d_exp", "hds_head_exp"]
    base_path = "hidden_dims_exp"
    print("yes")
    for alg in algs:
        plt.figure(figsize=(18, 9.6))
        fig = plt.figure(1)
        for k in range(3): # 表示两种情况
            dir_path = base_path + '/gaan_exp/' + dir_out[k] + '/calculations'
            df = {}
            df[0] = {}
            df[1] = {}
            for data in datasets: 
                file_path = dir_path + '/' + alg + '_' + data + '.csv' # 这里只与alg和data相关
                if not os.path.exists(file_path):
                    continue
                df_t = pd.read_csv(file_path, index_col=0).T
                if df_t.empty:
                    df[0][data] = [np.nan] * len(xticklabels[k])
                    df[1][data] = [np.nan] * len(xticklabels[k])
                else:
                    df[0][data] = df_t[0].values.tolist() + [np.nan] * (len(xticklabels[k]) - len(df_t[0].values))
                    df[1][data] = df_t[1].values.tolist() + [np.nan] * (len(xticklabels[k]) - len(df_t[1].values))

            df[0] = pd.DataFrame(df[0])
            df[1] = pd.DataFrame(df[1])
            ax1 = plt.subplot(2, 3, k + 1)
            ax2 = plt.subplot(2, 3, k + 4, sharex=ax1)
            for i, ax in enumerate([ax1, ax2]):
                ax.set_title(labels[i])
                if log_y:
                    ax.set_yscale("symlog", basey=2)
                ax.set_ylabel('Training Time / Epoch (ms)')
                ax.set_xlabel(xlabel[k])
                ax.set_xticks(list(range(len(xticklabels[k]))))
                ax.set_xticklabels(xticklabels[k])
                markers = 'oD^sdp'
                for j, c in enumerate(df[i].columns):
                    df[i][c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
                ax.legend()
        fig.tight_layout() # 防止重叠
        print(base_path + '/' + file_prefix + alg + "." + file_type)
        fig.savefig(base_path + '/' + file_prefix + alg + "." + file_type)
        plt.close()


def pic_calculations_gat():
    labels = ['Vertex Calculation', 'Edge Calculation']
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_"
    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabel = {'gat': [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"],
              'gaan': [r"$d_{v}$ and $d_{a}$ (#Head=4)", r"#Head ($d_{v}$=32, $d_{a}$=32)"] 
             }
    dir_out = ["hds_head_dims_exp", "hds_heads_exp"]
    base_path = "hidden_dims_exp"
    for alg in algs:
        plt.figure(figsize=(12, 9.6))
        fig = plt.figure(1)
        for k in range(2): # 表示两种情况
            dir_path = base_path + '/gat_exp/' + dir_out[k] + '/calculations'
            df = {}
            df[0] = {}
            df[1] = {}
            for data in datasets: 
                file_path = dir_path + '/' + alg + '_' + data + '.csv' # 这里只与alg和data相关
                if not os.path.exists(file_path):
                    continue
                df_t = pd.read_csv(file_path, index_col=0).T
                if df_t.empty:
                    df[0][data] = [np.nan] * len(xticklabels[k])
                    df[1][data] = [np.nan] * len(xticklabels[k])
                else:
                    df[0][data] = df_t[0].values.tolist() + [np.nan] * (len(xticklabels[k]) - len(df_t[0].values))
                    df[1][data] = df_t[1].values.tolist() + [np.nan] * (len(xticklabels[k]) - len(df_t[1].values))

            df[0] = pd.DataFrame(df[0])
            df[1] = pd.DataFrame(df[1])
            ax1 = plt.subplot(2, 2, 2 * k + 1)
            ax2 = plt.subplot(2, 2, 2 * k + 2, sharey=ax1)
            for i, ax in enumerate([ax1, ax2]):
                ax.set_title(labels[i])
                if log_y:
                    ax.set_yscale("symlog", basey=2)
                ax.set_ylabel('Training Time / Epoch (ms)')
                ax.set_xlabel(xlabel[alg][k])
                ax.set_xticks(list(range(len(xticklabels[k]))))
                ax.set_xticklabels(xticklabels[k])
                markers = 'oD^sdp'
                for j, c in enumerate(df[i].columns):
                    df[i][c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
                ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(base_path + "/" + file_prefix + alg + ".pdf")
        plt.close()
        

def preprocess2percent():
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = "layer_exp/calculations"
    for alg in algs:
        for data in datasets:
            csv_path = dir_out + '/' + alg + '_' + data
            df = pd.read_csv(csv_path + '.csv', index_col=0)
            pd.DataFrame(data=100 * df.values / df.values.sum(axis=0), index=df.index, columns=df.columns).to_csv(csv_path + '_precent.csv')
            

def pic_calculations_layer():
    # for GCN and GGNN
    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_proportion_"
    xticklabels = ['2', '3', '4', '5', '6', '7']
    xlabel = "Layer Depth"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = "layer_exp"
    labels = ['Vertex Calculation', 'Edge Calculation']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    dir_path = dir_out + '/calculations'
    for alg in algs:
        # plt.figure(figsize=(12, 4.8))
        fig = plt.figure(1)
        df = {}
        df[0] = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_path + '/' + alg + '_' + data + '_precent.csv' # 这里只与alg和data相关
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
            ax.set_title(labels[i])
            ax.set_ylabel('Proportion (%)')
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(xticklabels))))
            ax.set_xticklabels(xticklabels)
            markers = 'oD^sdp'
            for j, c in enumerate(df[i].columns):
                df[i][c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
            ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + ".png")
        plt.close()

def pic_calculations_layer_edge():
    # for GCN and GGNN
    file_prefix = "exp_layer_depth_on_vertex_edge_phase_proportion_"
    xticklabels = ['2', '3', '4', '5', '6', '7']
    xlabel = "Layer Depth"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = "layer_exp"
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    dir_path = dir_out + '/calculations'
    for alg in algs:
        plt.figure(figsize=(12, 4.8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        df = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_path + '/' + alg + '_' + data + '_precent.csv' # 这里只与alg和data相关
            if not os.path.exists(file_path):
                continue
            df_t = pd.read_csv(file_path, index_col=0).T
            if df_t.empty:
                df[1][data] = [np.nan] * len(xticklabels)
            else:
                df[1][data] = df_t[1].values.tolist() + [np.nan] * (len(xticklabels) - len(df_t[1].values))

        df[1] = pd.DataFrame(df[1])
        fig, ax = plt.subplots()
        ax.set_title(labels[1])
        ax.set_ylabel('Proportion (%)')
        ax.set_xlabel(xlabel)
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels)
        markers = 'oD^sdp'
        for j, c in enumerate(df[1].columns):
            df[1][c].plot(ax=ax, marker=markers[j], label=datasets_maps[c], rot=0)
        ax.legend()
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + ".png")
        plt.close()


def pic_calculations_avg_degree():
    file_prefix = "exp_avg_degree_on_vertex_edge_cal_time_"
    xticklabels = [3, 6, 10, 15, 20, 25, 30, 50]
    xlabel = "Average Vertex Degree"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = "exp_avg_degrees"
    labels = ['Vertex Calculation', 'Edge Calculation']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

    dir_path = dir_out + '/calculations'
    for alg in algs:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        df = pd.read_csv(dir_path + '/' + alg + '_graph.csv', index_col=0).T
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Training Time / Epoch (ms)")
        xticks = xticklabels[:len(df.index)]
        line1, = ax.plot(xticks, df[0].values, 'ob', label=labels[0], linestyle='-')
        line2, = ax.plot(xticks, df[1].values, 'Dg', label=labels[1], linestyle='-')
        
        edge_percent = df[1].values * 100 / df.values.sum(axis=1)
        ax2 = ax.twinx()
        ax2.set_ylabel("Proportion of Edge Calculation(%)")
        line3, = ax2.plot(xticks, edge_percent, 'rs--', label='Edge Calculation Percent')
        plt.legend(handles=[line1, line2, line3], loc="upper left")
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + ".pdf")
        plt.close()

pic_calculations_gaan(file_type="png")