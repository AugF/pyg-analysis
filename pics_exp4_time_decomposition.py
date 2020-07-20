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
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

# sampling
def pics_stages_bar(file_type="png"):
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp4_relative_sampling/batch_train_time_stack"
    modes = ['cluster', 'graphsage']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = 'paper_exp4_relative_sampling'
    file_out = 'exp_sampling_time_decomposition_'
    xticklabels = ['Sampling', 'Data Transferring', 'Training']
    
    for mode in modes:
        df_mean = {}
        df_min = {}
        df_max = {}
        for alg in algs:
            df_mean[alg] = [0.0] * 3
            df_min[alg] = [sys.maxsize] * 3
            df_max[alg] = [0.0] * 3
            cnt = 0
            for data in datasets:
                file_path = dir_path + alg + '_' + data + '_' + mode + '.log'
                if not os.path.exists(file_path):
                    continue
                print(file_path)
                sampling_time, to_time, train_time = 0, 0, 0
                with open(file_path) as f:
                    for line in f:
                        matchs_line = re.match(r".*, avg_epoch_train_time: (.*), avg_epochs_sampling_time:(.*), avg_epoch_to_time: (.*)", line)
                        if matchs_line:
                            sampling_time, to_time, train_time = matchs_line.group(2), matchs_line.group(3), matchs_line.group(1)
                            break
                #print(sampling_time, to_time, train_time)
                if sampling_time == 0 and to_time == 0 and train_time == 0:
                    continue
                cnt += 1
                all_time = [float(x) for x in [sampling_time, to_time, train_time]]
                sum_time = sum(all_time)
                all_time = [100 * x / sum_time for x in all_time]
                
                for j, x in enumerate(all_time):
                    df_mean[alg][j] += x
                    df_min[alg][j] = min(df_min[alg][j], x)
                    df_max[alg][j] = max(df_max[alg][j], x)

            for j in range(3):
                df_mean[alg][j] /= cnt
                df_min[alg][j] = df_mean[alg][j] - df_min[alg][j]
                df_max[alg][j] -= df_mean[alg][j]
             
        locations = [-1.5, -0.5, 0.5, 1.5]
        x = np.arange(len(xticklabels))
        width = 0.2
        rects = []
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
        fig, ax = plt.subplots()
        i = 0
        for (l, c) in zip(locations, colors):
            print(algs[i], df_mean[algs[i]], df_min[algs[i]], df_max[algs[i]])
            rects.append(ax.bar(x + l * width, df_mean[algs[i]], width, label=algorithms[algs[i]], color=c, yerr=[df_min[algs[i]], df_max[algs[i]]]))
            i += 1
        ax.set_ylabel("Proportion (%)")
        ax.set_xlabel("Stages")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.legend()
        fig.savefig(dir_out + file_out + mode + '.' + file_type)     