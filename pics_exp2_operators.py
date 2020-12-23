import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, datasets_maps, datasets
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def pics_operators_bar(dir_work="paper_exp2_time_break", file_out="exp_top_basic_ops_", file_type="png"):
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
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
                percent_ops = {k: 100.0 * ops[k] / s for k in ops.keys()}  # �������
                all_percent_ops[data] = percent_ops
                if res == {}:
                    res = percent_ops.copy()
                else:
                    for k in res.keys():
                        res[k] += percent_ops[k]

        res = {k: res[k] / cnt for k in res.keys()}  # �����ݼ���ƽ��
        res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)  # ����ѡ��topk����
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
        
        mean_values = df.values.mean(axis=1)
        max_values = df.values.max(axis=1) - mean_values
        min_values = mean_values - df.values.min(axis=1)
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Basic Operators")
        ax.set_ylabel("Proportion (%)")
        plt.bar(columns, mean_values, yerr=[min_values, max_values])
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(dir_work + '/' + file_out + alg + '.' + file_type)
        plt.close()
        
pics_operators_bar(dir_work="paper_exp5_inference_full", file_out="exp_inference_full_top_basic_ops_", file_type="png")
pics_operators_bar(dir_work="paper_exp5_inference_full", file_out="exp_inference_full_top_basic_ops_", file_type="pdf")