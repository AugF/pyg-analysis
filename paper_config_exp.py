# coding=utf-8
import os
import json
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, survey, algorithms, datasets_maps, datasets, dicts
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
dir_out = "config_exp"
algs = ['gcn', 'ggnn', 'gat', 'gaan']

def pics_epochs_violin():
    dir_out = "config_exp/epochs"
    for data in datasets:
        fig, ax = plt.subplots()
        ax.set_ylabel("Training Time / Epoch (ms)")
        ax.set_xlabel("Algorithm")
        df = pd.read_csv(dir_out + '/' + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax)
        plt.tight_layout()
        fig.savefig(dir_out + "/exp_absolute_training_time_comparison_" + data + ".png")

# survey: label='stages', 'layers', 'edge-cal', 'calculations'
def pic_others_propogation(label, file_name, file_type):
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    columns = dicts[label]
    for alg in algs:
        file_path = "config_exp/" + label + "/" + alg + ".csv"
        print(file_path)
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("Proportion (%)")
        ax.set_ylabel("Dataset")
        plt.tight_layout()
        fig.savefig("config_exp/" + label + "/" + file_name + alg + "." + file_type) 


def pics_operators():
    #plt.figure(figsize=(20, 15))
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #fig = plt.figure(1)  
    for i, alg in enumerate(algs):
        dir_path = dir_out + '/operators/' + alg + '_'
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

        df = {}  # ��ȡʵ�ʰٷֱ�
        for k in all_percent_ops.keys():
            df[k] = []
            for c in columns:
                df[k].append(all_percent_ops[k][c])
            df[k].append(100 - sum(df[k]))

        df = pd.DataFrame(df)
        df.to_csv(dir_out + "/operators/" + alg + ".csv")
        columns.append('others')
        
        fig, ax = survey([datasets_maps[d] for d in df.columns], df.values.T, columns)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("%")
        fig.tight_layout() # ��ֹ�ص� 
        fig.savefig(dir_out + '/operators/' + alg + '.png')
        plt.close()


def pics_operators_bar(file_type="png"):
    #plt.figure(figsize=(20, 15))
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #fig = plt.figure(1)  
    for i, alg in enumerate(algs):
        dir_path = dir_out + '/operators/' + alg + '_'
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

        df = {}  # ��ȡʵ�ʰٷֱ�
        for k in all_percent_ops.keys():
            df[k] = []
            for c in columns:
                df[k].append(all_percent_ops[k][c])
            df[k].append(100 - sum(df[k]))

        df = pd.DataFrame(df)
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
        fig.savefig(dir_out + '/operators/exp_top_basic_ops_' + alg + '.' + file_type)
        plt.close()


def pics_memory(file_type="png"):
    dir_name = r"/data/wangzhaokang/wangyunpan/pyg-gnns/config_exp/dir_json"
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Loss', 'Backward',
                   'Eval\nLayer0', 'Eval\nLayer1']

    #plt.figure(figsize=(12, 8))
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #fig = plt.figure(1)  
    for i, data in enumerate(datasets):
        allocated_max = {}
        for alg in algs:
            file_path = dir_name + '/config0_' + alg + '_' + data + '.json'
            print(file_path)
            if not os.path.exists(file_path):
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
                all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                     backward_end, layer0_eval, layer1_eval])
                all_data /= 1024 * 1024
                all_data = all_data.T  # �õ����еĔ���
                allocated_max[algorithms[alg]] = all_data[1] # current

        allocated_max = pd.DataFrame(allocated_max, index=time_labels)

        #ax = plt.subplot(2, 3, i + 1)
        fig, ax = plt.subplots()
        ax.set_ylabel("Peak Memory Usage (MB)")
        ax.set_xlabel("Stage")
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        ax.set_xticks(list(range(len(time_labels))))
        ax.set_xticklabels(time_labels)
        for j, c in enumerate(allocated_max.columns):
            allocated_max[c].plot(ax=ax, marker=markers[j], linestyle=lines[j], label=c, rot=45)
        ax.legend()
        fig.tight_layout() 
        fig.savefig(dir_out + '/memory/exp_memory_usage_stage_' + datasets_maps[data] + '.' + file_type)
        plt.close()

# pics_operators_bar(file_type="png")
# pics_memory(file_type="pdf")
# pics_epochs_violin()
# pic_others_propogation('calculations', 'exp_vertex_edge_cal_proportion_', "pdf")
# pics_operators()
