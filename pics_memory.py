import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def run_memory_file(params):
    dir_memory, dir_out, algs, datasets = params['dir_memory'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    base_path = os.path.join(dir_out, "memory")
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
                    df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
            if df[data] == [None] * (len(variables)):
                del df[data]
        df = pd.DataFrame(df)
        df.to_csv(base_path + "/" + alg + ".csv")


def run_memory_gcn_ggnn():
    file_out="exp_hyperparameter_on_memory_usage_"
    params = yaml.load(open("cfg_file/hds_exp.yaml"))
    
    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_json"
    dir_out = "hidden_dims_exp"
    base_path = os.path.join(dir_out, "hds_exp", "memory")
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
                    df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
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
            df[c].plot(ax=ax, marker=markers[i], label=c, rot=0)
        ax.legend()
        df.to_csv(base_path + '/' + alg + ".csv")
        fig.savefig(dir_out + "/" + file_out + alg + ".png")
        plt.close()


def run_memory_gat():
    file_out="exp_hyperparameter_on_memory_usage_"
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    file_prefix = ['_4_', '_']
    file_suffix = ['', '_32']
    xlabel = {'gat': [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"],
              'gaan': [r"$d_{v}$ and #$d_{a}$(#Head=4)", r"#Head ($d_{v}$=32, $d_{a}$=32)"] 
             }
    
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_head_json"
    dir_out = "hidden_dims_exp"
    dir_path = ['hds_head_dims_exp', 'hds_heads_exp']
    for alg in algs:
        plt.figure(figsize=(12, 4.8))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig = plt.figure(1)
        for k in range(2):
            base_path = os.path.join(dir_out, dir_path[k], "memory")
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
            ax = plt.subplot(1, 2, k + 1)
            # ax.set_title("GPU Memory Usage")
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel("Training Memory Usage (MB)")
            ax.set_xlabel(xlabel[alg][k])
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels([str(i) for i in xticklabels[k]])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=c, rot=0)
            ax.legend()
            df.to_csv(base_path + '/' + alg + ".csv")
        fig.savefig(dir_out + "/" + file_out + alg + ".png")
        plt.close()

def run_memory_layer():
    file_out="exp_hyperparameter_on_memory_usage_"
    params = yaml.load(open("cfg_file/layer_exp.yaml"))
    
    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_json"
    dir_out = "layer_exp"
    base_path = os.path.join(dir_out, "memory")
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
                    df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
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
            df[c].plot(ax=ax, marker=markers[i], label=c, rot=0)
        ax.legend()
        df.to_csv(base_path + '/' + alg + ".csv")
        fig.savefig(dir_out + "/" + file_out + alg + ".png")
        plt.close()