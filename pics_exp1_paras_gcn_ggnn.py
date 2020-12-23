import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, datasets_maps
plt.style.use("ggplot")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


def pic_calculations_gcn_ggnn(file_type):
    plt.rcParams["font.size"] = 12
    xlabel = "Dimension of Hidden Vectors\n" + r"($dim(\mathbf{h}^1_x)$)"
    # for GCN and GGNN
    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_"
    xticklabels = ['16', '32', '64', '128', '256', '512', '1024', '2048']
    algs = ['gcn', 'ggnn']
    labels = ['Vertex Calculation', 'Edge Calculation']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers',
                'coauthor-physics', 'flickr', 'com-amazon']

    dir_in = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters"
    dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters/paras_fig"
    for alg in algs:
        fig, axes = plt.subplots(1, 2, figsize=(
            7, 7/2), sharey=True, tight_layout=True)
        df = {}
        df[0] = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_in + '/hidden_dims_exp/hds_exp/calculations/' + \
                alg + '_' + data + '.csv'  # 这里只与alg和data相关

            if not os.path.exists(file_path):
                continue
            df_t = pd.read_csv(file_path, index_col=0).T
            if df_t.empty:
                df[0][data] = [np.nan] * len(xticklabels)
                df[1][data] = [np.nan] * len(xticklabels)
            else:
                df[0][data] = df_t[0].values.tolist() + [np.nan] * \
                    (len(xticklabels) - len(df_t[0].values))
                df[1][data] = df_t[1].values.tolist() + [np.nan] * \
                    (len(xticklabels) - len(df_t[1].values))

        df[0] = pd.DataFrame(df[0])
        df[1] = pd.DataFrame(df[1])
        print(df[0], df[1])
        for i in range(2):
            ax = axes[i]
            ax.set_title(labels[i], fontsize=12)
            ax.set_yscale("symlog", basey=2)
            if i == 0:
                ax.set_ylabel('Training Time / Epoch (ms)', fontsize=12)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks(list(range(len(xticklabels))))
            ax.set_xticklabels(xticklabels, fontsize=10, rotation=30)
            markers = 'oD^sdp'
            for j, c in enumerate(df[i].columns):
                ax.plot(df[i].index, df[i][c], marker=markers[j],
                        label=datasets_maps[c])
            ax.legend()
        fig.savefig(dir_out + "/" + file_prefix + alg + "." + file_type)
        plt.close()


def run_memory_gcn_ggnn(file_type):
    file_out = "exp_hyperparameter_on_memory_usage_"
    params = yaml.load(open("cfg_file/gcn_ggnn_exp_hds.yaml"))

    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, log_y = params['variables'], params[
        'file_prefix'], params['file_suffix'], params['log_y']
    xlabel = "Dimension of Hidden Vectors\n" + r"($dim(\mathbf{h}^1_x)$)"

    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp1_super_parameters/dir_gcn_ggnn_json"
    dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters/paras_fig"

    for alg in algs:
        df = {}
        for data in datasets:
            df[data] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + \
                    data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df[data].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(
                        res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(
                        res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                         backward_end, layer0_eval, layer1_eval, eval_end])
                    all_data /= (1024 * 1024)
                    # 这里记录allocated_bytes.all.max
                    df[data].append(max(all_data[2:, 1]) - all_data[0, 0])

            if df[data] == [None] * (len(variables)):
                del df[data]
        df = pd.DataFrame(df)
        df.to_csv(dir_out + "/" + alg + ".csv")
        fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
        ax.set_yscale("symlog", basey=2)
        ax.set_ylabel("Training Memory Usage (MB)", fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_xticks(list(range(len(variables))))
        ax.set_xticklabels([str(i) for i in variables], fontsize=10, rotation=30)
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            ax.plot(df.index, df[c], marker=markers[i], label=datasets_maps[c])
        ax.legend()
        fig.savefig(dir_out + "/" + file_out + alg + "." + file_type)
        plt.close()


pic_calculations_gcn_ggnn(file_type="png")
pic_calculations_gcn_ggnn(file_type="pdf")
run_memory_gcn_ggnn(file_type="png")
run_memory_gcn_ggnn(file_type="pdf")
