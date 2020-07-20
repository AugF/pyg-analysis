import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, autolabel, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def pic_calculations_gaan(file_type="png"):
    labels = ['Vertex Calculation', 'Edge Calculation']
    algs = ['gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_"
    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], ['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabel = [r"Hidden Vector Dimension $dim(\mathbf{h}^1)$ (#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$(#Head=4, $dim(\mathbf{h}^1)$=64)",  r"#Head ($dim(\mathbf{h}^1)$=64, $d_a=d_v=d_m$=32)"]
    dir_paths = ["hds", "hds_d", "hds_head"]
    dir_in = "paper_exp1_super_parameters"
    dir_out = "paper_exp1_super_parameters"
    
    for alg in algs:
        plt.figure(figsize=(18, 9.6))
        fig = plt.figure(1)
        for k in range(3): # 表示两种情况
            dir_path = dir_in + '/gaan_exp/' + dir_paths[k] + '/calculations'
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
        print(dir_out + '/' + file_prefix + alg + "." + file_type)
        fig.savefig(dir_out + '/' + file_prefix + alg + "." + file_type)
        plt.close()


def run_memory_gaan(file_type="png"):
    file_out="exp_hyperparameter_on_memory_usage_"
    algs = ['gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], ['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabel = [r"Hidden Vector Dimension $dim(\mathbf{h}^1)$ (#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$(#Head=4, $dim(\mathbf{h}^1)$=64)",  r"#Head ($dim(\mathbf{h}^1)$=64, $d_a=d_v=d_m$=32)"]
    
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp1_super_parameters/dir_gaan_json"
    dir_out = "paper_exp1_super_parameters"
    dir_paths = ["hds", "hds_d", "hds_head"]
    
    file_prefix = ['_4_32_', '_4_', '_']
    file_suffix = ['', '_64', '_32_64']
    
    for alg in algs:
        plt.figure(figsize=(18, 4.8))
        fig = plt.figure(1)
        for k in range(3):
            base_path = os.path.join(dir_out, 'gaan_exp', dir_paths[k], "memory")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            df = {}
            for data in datasets:
                df[data] = []
                for var in xticklabels[k]:
                    file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix[k] + str(var) + file_suffix[k] + '.json'
                    if not os.path.exists(file_path):
                        df[data].append(None)
                        continue
                    with open(file_path) as f:
                        res = json.load(f)
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
                        df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
                if df[data] == [None] * (len(xticklabels[k])):
                    del df[data]
            df = pd.DataFrame(df)
            ax = plt.subplot(1, 3, k + 1)
            # ax.set_title("GPU Memory Usage")
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel("Training Memory Usage (MB)")
            ax.set_xlabel(xlabel[k])
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels([str(i) for i in xticklabels[k]])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=datasets_maps[c], rot=0)
            ax.legend()
            df.to_csv(base_path + '/' + alg + ".csv")
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_out + alg + "." + file_type)
        plt.close()

run_memory_gaan()
run_memory_gaan("pdf")