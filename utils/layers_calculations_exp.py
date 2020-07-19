import os
import time
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, all_labels, survey, algorithms, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def get_layers_calculations_time(cur, outliers, alg, layer=2):
    labels = all_labels[alg]
    layers_cal_times = [0] * 2 * layer
    for label in labels:
        sql = "select start, end, text from nvtx_events where text == '{}'".format(label)
        res = cur.execute(sql).fetchall()[layer:]  # 不考虑warm up
        cost_time = 0
        for i in range(50):
            if i in outliers: continue
            # epoch_time = forward time + backward time + eval time
            layers_times = [0] * layer
            # 1. 获取forward time和eval time
            for j in range(layer):
                layers_times[j] += get_real_time(res[2 * layer * i + j], cur)[0] + get_real_time(res[2 * layer * i + j + layer], cur)[0]
            # 2. 基于forward的标签对应的seq获取backward time
            for j in range(layer):
                # 思路：首先得到label的时间段[st, ed]; 然后寻找该时间段中所有的seq, 然后找对应的backward中的seq
                # 2.1 寻找该时间段中所有的seq
                seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                seq_res = cur.execute(seq_sql.format(res[2 * layer * i + j][0], res[2 * layer * i + j][1])).fetchall()

                min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                if start_time:
                    backward_time = get_real_time((start_time[1], end_time[1], label), cur)[0]
                else:
                    start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]
                layers_times[j] += backward_time

        if 'vertex' in label:
            for j in range(layer):
                layers_cal_times[2 * j] += layers_times[j] / (50 - len(outliers))
        else:
            for j in range(layer):
                layers_cal_times[2 * j + 1] += layers_times[j] / (50 - len(outliers))
    return layers_cal_times

# if len(sys.argv) < 2 or sys.argv[1] not in algs:
#     print("python calculations_exp.py gcn")
#     sys.exit(0)

# alg = sys.argv[1]

def run_layers_calculations_exp(params):
    dir_name, dir_out, algs, datasets = params['dir_name'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix = params['variables'], params['file_prefix'], params['file_suffix']

    layer_var_flag = params['dir_out'] == 'layer_exp'
    base_path = os.path.join(dir_out, "calculations")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for alg in algs:
        for data in datasets:
            out_path = base_path + '/' + alg + '_' + data + '.csv'
            if os.path.exists(out_path):
                continue
            df = {}
            for var in variables:
                outlier_file = dir_out + '/epochs/' + alg + '_' + data + file_prefix + str(var) + file_suffix + '_outliers.txt'
                file_path = dir_name + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.sqlite'
                if not os.path.exists(file_path) or not os.path.exists(outlier_file):
                    continue
                print(file_path)
                cur = sqlite3.connect(file_path).cursor()
                print(data, alg)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
                if layer_var_flag:
                    layer = var
                else: layer = params['layer']
                res = get_layers_calculations_time(cur, outliers, alg, layer=layer) # layer实验特有
                df[var] = res
            pd.DataFrame(df).to_csv(out_path)


def run_one_file():
    import yaml
    params = yaml.load(open('cfg_file/hidden_dims_3_exp.yaml'))
    dir_name, dir_out, algs, datasets = params['dir_name'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix = params['variables'], params['file_prefix'], params['file_suffix']

    alg, data, var = 'gcn', 'amazon-photo', 16
    base_path = os.path.join(dir_out, "calculations")
    csv_path = base_path + '/' + alg + '_' + data + '.csv'
    outlier_file = dir_out + '/epochs/' + alg + '_' + data + file_prefix + str(var) + file_suffix + '_outliers.txt'
    file_path = dir_name + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.sqlite'
    if os.path.exists(file_path):
        cur = sqlite3.connect(file_path).cursor()
        print(data, alg)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
        res = get_layes_calculations_time(cur, outliers, alg, 3)
        print(res)

def run_config_exp():
    dir_out = "config_exp"
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/config_exp/dir_sqlite"
    base_path = os.path.join(dir_out, "layers_calculations")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    for alg in algs:
        df = {}
        out_path = base_path + '/' + alg + '.csv'
        if os.path.exists(out_path):
            continue
        for data in datasets:
            outlier_file = 'config_exp/epochs/' + alg + '_' + data + '_outliers.txt'
            file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
            if not os.path.exists(file_path) or not os.path.exists(outlier_file):
                continue
            cur = sqlite3.connect(file_path).cursor()
            print(data, alg)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
            res = get_layers_calculations_time(cur, outliers, alg)
            df[data] = res
        pd.DataFrame(df).to_csv(out_path)


def run_layer_exp_layer3():
    dir_out = "config_exp"
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_name = "/data/wangzhaokang/wangyunpan/pyg-gnns/layer_exp/dir_sqlite"
    base_path = os.path.join(dir_out, "layers_calculations")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    for alg in algs:
        df = {}
        out_path = base_path + '/' + alg + '_3.csv'
        if os.path.exists(out_path):
            continue
        for data in datasets:
            outlier_file = 'layer_exp/epochs/' + alg + '_' + data + '_3_outliers.txt'
            file_path = dir_name + '/config0_' + alg + '_' + data + '_3.sqlite'
            if not os.path.exists(file_path) or not os.path.exists(outlier_file):
                continue
            cur = sqlite3.connect(file_path).cursor()
            print(data, alg)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
            res = get_layers_calculations_time(cur, outliers, alg, 3)
            df[data] = res
        pd.DataFrame(df).to_csv(out_path)

def pic_config_exp_layer2(file_type="png"):
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    columns = ['Layer0-Vertex', 'Layer0-Edge', 'Layer1-Vertex', 'Layer1-Edge']
    for alg in algs:
        file_path = "config_exp/layers_calculations/" + alg + ".csv"
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("Proportion (%)")
        ax.set_ylabel("Dataset")
        plt.tight_layout()
        fig.savefig("config_exp/layers_calculations/exp_layer_time_proportion_" + alg + "." + file_type) 


def pic_config_exp_layer3():
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    columns = ['Layer0-V', 'Layer0-E', 'Layer1-V', 'Layer1-E', 'Layer2-V', 'Layer2-E']
    for alg in algs:
        file_path = "config_exp/layers_calculations/" + alg + "_3.csv"
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("Proportion (%)")
        ax.set_ylabel("Dataset")
        plt.tight_layout()
        fig.savefig("config_exp/layers_calculations/exp_layer3_time_proportion_" + alg + ".png") 


pic_config_exp_layer2("pdf")
