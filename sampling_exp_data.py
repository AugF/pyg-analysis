import json
import sys
import re
# import argparse
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, algorithms, sampling_modes, survey
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

# parser = argparse.ArgumentParser()
# parser.add_argument('--dir_name', type=str, default='dir_sparse_features_memory')
# parser.add_argument('--dir_out', type=str, default='sparse_exp')
# parser.add_argument('--vars', type=str, default='16 32 64 128 256 512')
# args = parser.parse_args()

# dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/data_exp/" + args.dir_name
# dir_out = "sampling_exp/data_exp/" + args.dir_out
# variables = args.vars.split(' ')

modes = ['graphsage', 'cluster']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']


def run_time(dir_name, dir_out, variables, file_prefix, file_suffix, datasets, xlabel):
    time_path = os.path.join(dir_out, "time")
    if not os.path.exists(time_path):
        os.makedirs(time_path)
    for mode in modes:
        for alg in algs:
            df = {}
            for data in datasets:
                df[data] = []
                for var in variables:
                    file_path = dir_name + "/" + mode + '_' + alg + "_" + data + file_prefix + str(var)+ file_suffix + ".log"
                    if not os.path.exists(file_path):
                        df[data].append(np.nan)
                        continue
                    print(file_path)
                    my_file = open(file_path)
                    all_time = 0.0
                    for line in my_file:
                        match_title = re.match(r".*all_time: (.*), loss:.*", line)
                        if match_title:
                            all_time += float(match_title.group(1))
                    my_file.close()
                    df[data].append(all_time / 50)
                if df[data] == [np.nan] * len(variables):
                    del df[data]
            df = pd.DataFrame(df)
            df.to_csv(time_path + '/time_' + mode + '_' + alg + '.csv')
            fig, ax = plt.subplots()
            ax.set_title(sampling_modes[mode] + ' ' + algorithms[alg])
            ax.set_ylabel("Training Time /Epoch (s)")
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(variables))))
            ax.set_xticklabels(variables)
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=c)
            ax.legend()
            fig.savefig(time_path + "/time_" + mode + '_' + alg +  ".png")
            plt.close()


def run_memory(dir_name, dir_out, variables, file_prefix, file_suffix, datasets, xlabel):
    memory_path = os.path.join(dir_out, "memory")
    if not os.path.exists(memory_path):
        os.makedirs(memory_path)
    for mode in modes:
        for alg in algs:
            df = {}
            for data in datasets:
                df[data] = []
                for var in variables:
                    file_path = dir_name + "/" + mode + '_' + alg + "_" + data + file_prefix + str(var)+ file_suffix + ".json"
                    # print(file_path)
                    if not os.path.exists(file_path):
                        df[data].append(np.nan)
                        continue
                    # print(file_path)
                    with open(file_path) as f:
                        res = json.load(f)
                        warmup_end = np.array(res['warmup end']).mean(axis=0)
                        layer0 = np.array(res['layer0'][1:]).mean(axis=0)
                        layer1 = np.array(res['layer1'][1:]).mean(axis=0)
                        forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                        backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                        all_data = np.array([warmup_end, layer0, layer1, forward_end,
                                            backward_end])
                        all_data /= (1024 * 1024)
                        df[data].append(max(all_data[:, 1]) - all_data[0, 1]) # 这里记录allocated_bytes.all.max
                if df[data] == [np.nan] * len(variables):
                    del df[data]
            df = pd.DataFrame(df)
            df.to_csv(memory_path + '/memory_' + mode + '_' + alg + '.csv')
            fig, ax = plt.subplots()
            ax.set_title(sampling_modes[mode] + ' ' + algorithms[alg])
            ax.set_ylabel("GPU Memory Usage(MB)")
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(variables))))
            ax.set_xticklabels(variables)
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=c, rot=10)
            ax.legend()
            fig.savefig(memory_path + "/memory_" + mode + '_' + alg +  ".png")
            plt.close()
            
            
dir_names = ['dir_dense', 'dir_sparse_features', 'dir_sparse_features', 'dir_graph', 'dir_graph']
dir_outs = ['dense_exp', 'sparse_dims_exp', 'sparse_ratios_exp', 'graph_nodes_exp', 'graph_degrees_exp']
variables = ['16 32 64 128 256 512', '250 500 750 1000 1250', '5 10 20 50', '25 50 75 100 250 750 1000', '2 5 10 25 50 75 100']
xlabels = ['Dense Feats Dims', 'Sparse Feats Dims', 'Sparse Feats Zeros Ratio', 'Graph Nodes(k)', 'Graph Degrees']
file_prefixs = ['_', '_', '_500_', '_', '_50k_']
file_suffixs = ['', '_20', '', 'k_25', '']

for i in range(5):
    dir_name =  "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/data_exp/" + dir_names[i]
    dir_out = "sampling_exp/data_exp/" + dir_outs[i]
    variable = [int(i) for i in variables[i].split(' ')]
    if i >= 3: 
        run_memory(dir_out=dir_out, dir_name=dir_name + '_memory', variables=variable, file_prefix=file_prefixs[i], file_suffix=file_suffixs[i], datasets=['graph'], xlabel=xlabels[i])
        run_time(dir_out=dir_out, dir_name=dir_name + '_time', variables=variable, file_prefix=file_prefixs[i], file_suffix=file_suffixs[i], datasets=['graph'], xlabel=xlabels[i])
    else:
        run_memory(dir_out=dir_out, dir_name=dir_name + '_memory', variables=variable, file_prefix=file_prefixs[i], file_suffix=file_suffixs[i], datasets=datasets, xlabel=xlabels[i])
        run_time(dir_out=dir_out, dir_name=dir_name + '_time', variables=variable, file_prefix=file_prefixs[i], file_suffix=file_suffixs[i], datasets=datasets, xlabel=xlabels[i])
