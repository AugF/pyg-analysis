import json
import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, algorithms, sampling_modes, survey
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12


dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/batch_exp/dir_batch"
dir_out = "sampling_exp/batch_exp"
modes = ['graphsage', 'cluster']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    
memory_path = os.path.join(dir_out, "memory")
if not os.path.exists(memory_path):
    os.makedirs(memory_path)

# memory experiment
def run_memory():
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/batch_exp/dir_batch"
    variables = {
        'graphsage': [128, 256, 512, 1024, 2048],
        'cluster': [1, 2, 4, 8, 16, 32, 64],
    }
    xlables = {
        'graphsage': 'Batch Size',
        'cluster': 'Batch Cluster Size' # 1500
    }
    for mode in modes:
        for alg in algs:
            df = {}
            alg_path = memory_path + '/memory_' + mode + '_' + alg + '.csv'
            if os.path.exists(alg_path):
                continue
            for data in datasets:
                df[data] = []
                for var in variables[mode]:
                    file_path = dir_name + "/" + mode + '_' + alg + "_" + data + "_" + str(var)+ ".json"
                    if not os.path.exists(file_path):
                        df[data].append(np.nan)
                        continue
                    print(file_path)
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
            df = pd.DataFrame(df)
            df.to_csv(alg_path)
            fig, ax = plt.subplots()
            ax.set_title(sampling_modes[mode] + ' ' + algorithms[alg])
            ax.set_ylabel("GPU Memory Usage(MB)")
            ax.set_xlabel(xlables[mode])
            ax.set_xticks(list(range(len(variables[mode]))))
            ax.set_xticklabels(variables[mode])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=c, rot=10)
            ax.legend()
            fig.savefig(memory_path + "/memory_" + mode + '_' + alg +  ".png")
            plt.close()

run_memory()
