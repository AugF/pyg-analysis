import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algorithms, variables, datasets_maps, datasets
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 12

def pics_operators_bar(dir_out="exp3_thesis_figs/time", dir_work="paper_exp2_time_break",
                        file_out="exp_top_basic_ops_", file_type="png"):
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    # plt.rcParams["font.size"] = 12
    for i, alg in enumerate(algs):
        dir_path = dir_work + '/config_exp/operators/' + alg + '_'
        all_percent_ops = {}  #
        res = {}
        cnt = 0
        for data in datasets:
            file_path = dir_path + data + '.json'
            if not os.path.exists(file_path):
                continue
            cnt += 1
            with open(file_path) as f:
                ops = json.load(f)
                s = sum(ops.values())
                percent_ops = {k: 100.0 * ops[k] / s for k in ops.keys()}  
                all_percent_ops[data] = percent_ops
                if res == {}:
                    res = percent_ops.copy()
                else:
                    for k in res.keys():
                        res[k] += percent_ops[k]

        res = {k: res[k] / cnt for k in res.keys()} 
        res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)  
        columns = [i[0] for i in res_sort[:5]]

        df = {} 
        for k in all_percent_ops.keys():
            df[k] = []
            for c in columns:
                df[k].append(all_percent_ops[k][c])
            df[k].append(100 - sum(df[k]))

        df = pd.DataFrame(df)
        df.to_csv(dir_work + "/config_exp/operators/" + alg + ".csv")
        columns.append('others')

        for i, c in enumerate(columns):
            if c == '_thnn_fused_gru_cell':
                columns[i] = '_thnn_fused_' + '\n' + 'gru_cell'
                        
        mean_values = df.values.mean(axis=1)
        max_values = df.values.max(axis=1) - mean_values
        min_values = mean_values - df.values.min(axis=1)
        
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
        ax.set_xlabel("基础算子", fontsize=base_size+2)
        ax.set_ylabel("比例 (%)", fontsize=base_size+2)
        plt.bar(columns, mean_values, yerr=[min_values, max_values], color='red')
        plt.xticks(rotation=20, fontsize=base_size-1)
        plt.yticks(fontsize=base_size)
        fig.savefig(dir_out + '/' + file_out + alg + '.' + file_type, dpi=400)
        plt.close()
        

pics_operators_bar()
pics_operators_bar(dir_out="exp3_thesis_figs/time", dir_work="paper_exp5_inference_full",
                   file_out="exp_inference_full_top_basic_ops_", file_type="png")
