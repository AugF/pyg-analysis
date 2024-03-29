import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, autolabel, datasets_maps
from matplotlib.font_manager import _rebuild
_rebuild() 
base_size = 12
plt.rcParams["font.size"] = base_size
plt.rcParams["font.size"] = 12

def run_memory_degrees(file_type="png", dir_save="./"):
    file_out="exp_memory_expansion_ratio_input_graph_number_of_edges_"
    log_y = False    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['graph']
    
    variables = [2, 5, 10, 15, 20, 30, 40, 50, 70]
    xticklabels = variables
    
    dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp3_memory/dir_degrees_json"
    base_path = "paper_exp3_memory"
    xlabel = ""

    file_prefix, file_suffix = '_10k_', ''
    
    df_peak = {}
    df_ratio = {}
    df_current = {}
    for alg in algs:
        df_peak[alg] = []
        df_ratio[alg] = []
        df_current[alg] = []
        for data in datasets:
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data  + file_prefix + str(var) + file_suffix + '.json'
                print(file_path)
                if not os.path.exists(file_path):
                    df_peak[alg].append(np.nan)
                    df_ratio[alg].append(np.nan)
                    df_current[alg].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    print(file_path)
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
                    
                    all_data /= (1024 * 1024 * 1024)
                    df_peak[alg].append(max(all_data[2:, 1]))
                    df_ratio[alg].append(max(all_data[2:, 1]) / all_data[0, 0]) # 这里记录allocated_bytes.all.peak
                    df_current[alg].append(all_data[0, 0])
    
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel("Peak Memory Usage (GB)", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    df_peak = pd.DataFrame(df_peak)
    markers = 'oD^sdp'
    for i, c in enumerate(df_peak.columns):
        ax.plot(xticklabels, df_peak[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=12)
    fig.savefig(dir_save + "/" + file_out + "peak_memory." + file_type)
    
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel("Expansion Ratio", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    df_ratio = pd.DataFrame(df_ratio)
    markers = 'oD^sdp'
    for i, c in enumerate(df_ratio.columns):
        ax.plot(xticklabels, df_ratio[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=12)
    fig.savefig(dir_save + "/" + file_out + "expansion_ratio." + file_type)


def run_inference_full_memory_degrees(file_type="png", dir_save="paper_exp5_inference_full"):
    file_out="exp_inference_full_memory_expansion_ratio_input_graph_number_of_edges_"
    log_y = False    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['graph']
    
    variables = [2, 5, 10, 15, 20, 30, 40, 50, 70]
    xticklabels = variables
    
    dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp8_inference_full/dir_degrees_json"
    base_path = "paper_exp5_inference_full"
    xlabel = "Average Degree"

    file_prefix, file_suffix = '_10k_', ''
    
    df_peak = {}
    df_ratio = {}
    for alg in algs:
        df_peak[alg] = []
        df_ratio[alg] = []
        for data in datasets:
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data  + file_prefix + str(var) + file_suffix + '.json'
                print(file_path)
                if not os.path.exists(file_path):
                    df_peak[alg].append(np.nan)
                    df_ratio[alg].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    data_memory = res['data load'][0][0]
                    max_memory = 0
                    for k in res.keys():
                        max_memory = max(max_memory, np.array(res[k]).mean(axis=0)[1])
                    
                    print("data memory", data_memory)
                    print("max memory", max_memory / data_memory)
                    df_ratio[alg].append(max_memory / data_memory) # 这里记录allocated_bytes.all.peak
                    df_peak[alg].append(max_memory / (1024 * 1024 * 1024))
    
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel("Peak Memory Usage (GB)", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    df_peak = pd.DataFrame(df_peak)
    markers = 'oD^sdp'
    for i, c in enumerate(df_peak.columns):
        ax.plot(xticklabels, df_peak[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=12)
    fig.savefig(dir_save + "/" + file_out + "peak_memory." + file_type)
    
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel("Expansion Ratio", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    df_ratio = pd.DataFrame(df_ratio)
    markers = 'oD^sdp'
    for i, c in enumerate(df_ratio.columns):
        ax.plot(xticklabels, df_ratio[c], marker=markers[i], label=algorithms[c])
    ax.legend(fontsize=12)
    fig.savefig(dir_save + "/" + file_out + "expansion_ratio." + file_type)


run_memory_degrees(file_type="png", dir_save="paper_exp2_time_break/paper_figs")
# run_memory_degrees(file_type="pdf", dir_save="/mnt/data/wangzhaokang/wangyunpan/pyg-analysis/exp_supplement")
# run_inference_full_memory_degrees(file_type="png", dir_save="/mnt/data/wangzhaokang/wangyunpan/pyg-analysis/exp_supplement")
# run_inference_full_memory_degrees(file_type="pdf", dir_save="/mnt/data/wangzhaokang/wangyunpan/pyg-analysis/exp_supplement")