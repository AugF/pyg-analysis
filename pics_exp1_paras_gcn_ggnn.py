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

   
def run_memory_gcn_ggnn():
    file_out="exp_hyperparameter_on_memory_usage_"
    params = yaml.load(open("cfg_file/gcn_ggnn_exp_hds.yaml"))
    
    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/hidden_dims_exp/dir_json"
    dir_out = "hidden_dims_exp"
    base_path = os.path.join(dir_out, "gcn_ggnn_exp/hds_exp", "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for alg in algs:
        df = {}
        for data in datasets:
            df[data] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df[data].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    # print(file_path)
                    # print(res.keys())
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    all_data /= (1024 * 1024)
                    df[data].append(max(all_data[2:, 1]) - all_data[0, 0]) # 这里记录allocated_bytes.all.max
            
            if df[data] == [None] * (len(variables)):
                del df[data]
        df = pd.DataFrame(df)
        fig, ax = plt.subplots()
        # ax.set_title("GPU Memory Usage")
        if log_y:
            ax.set_yscale("symlog", basey=2)
        ax.set_ylabel("Training Memory Usage (MB)")
        ax.set_xlabel(xlabel)
        ax.set_xticks(list(range(len(variables))))
        ax.set_xticklabels([str(i) for i in variables])
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[i], label=datasets_maps[c], rot=0)
        ax.legend()
        df.to_csv(base_path + '/' + alg + ".csv")
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_out + alg + ".pdf")
        plt.close()