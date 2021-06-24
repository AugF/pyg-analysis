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

def run_memory_factors_dense_feats_dims(file_type="png"):
    file_out="exp_memory_expansion_ratio_input_feature_dimension_"
    log_y = True    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['com-amazon']
    variables = [16, 32, 64, 128, 256, 512]
    xticklabels = variables

    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp3_memory/dir_feat_dims_json"
    base_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/new_exp_supplement"
    
    file_prefix, file_suffix = '_', ''
    
    for data in datasets:
        df_ratio = {}
        for alg in algs:
            df_ratio[alg] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data  + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df_ratio[alg].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    print(file_path)
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
                    df_ratio[alg].append(max(all_data[2:, 1]) / all_data[0, 0]) # 这里记录allocated_bytes.all.peak
    
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_ylabel("Expansion Ratio", fontsize=16)
        ax.set_xlabel("Input Feature Dimension", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xticklabels([0] + xticklabels)
        df_ratio = pd.DataFrame(df_ratio)
        if log_y:
            ax.set_yscale("symlog", basey=2)
         
        locations = [-1.5, -0.5, 0.5, 1.5]
        x = np.arange(len(variables))
        width = 0.2
        rects = []
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
        
        i = 0
        for (col, c) in zip(df_ratio.columns, colors):
            rects.append(ax.bar(x + locations[i] * width, df_ratio[col], width, label=algorithms[algs[i]], color=c))
            i += 1
        ax.legend(fontsize=12)
        fig.savefig(base_path + "/" + file_out + data + "." + file_type)


def run_inference_full_memory_feats_dims(file_type="png"):
    file_out="exp_memory_expansion_ratio_input_feature_dimension_"
    log_y = True    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['com-amazon']
    variables = [16, 32, 64, 128, 256, 512]
    xticklabels = variables

    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp8_inference_full/dir_feat_dims_json"
    base_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/exp_supplement"
    
    file_prefix, file_suffix = '_', ''
    
    for data in datasets:
        df_ratio = {}
        df_peak = {}
        for alg in algs:
            df_ratio[alg] = []
            df_peak[alg] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data  + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df_ratio[alg].append(np.nan)
                    df_peak[alg].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    print(file_path)
                    data_memory = res['data load'][0][0]
                    max_memory = 0
                    for k in res.keys():
                        max_memory = max(max_memory, np.array(res[k]).mean(axis=0)[1])
                    
                    print("data memory", data_memory)
                    print("max memory", max_memory)
                    df_ratio[alg].append(max_memory / data_memory) # 这里记录allocated_bytes.all.peak
                    df_peak[alg].append(max_memory / (1024 * 1024))           
                    
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_ylabel("Peak Memory Usage (MB)", fontsize=16)
        ax.set_xlabel("Input Feature Dimension", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        df_peak = pd.DataFrame(df_peak)
        # df_peak.to_csv(base_path + "/" + file_out + data + "_peak_memory.csv")
        markers = 'oD^sdp'
        for i, c in enumerate(df_peak.columns):
            ax.plot(xticklabels, df_peak[c], marker=markers[i], label=algorithms[c])
        ax.legend(fontsize=12)
        fig.savefig(base_path + "/" + file_out + data + "_peak_memory." + file_type)
                    
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_ylabel("Expansion Ratio", fontsize=16)
        ax.set_xlabel("Input Feature Dimension", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_xticklabels([0] + xticklabels)
        df_ratio = pd.DataFrame(df_ratio)
        # df_ratio.to_csv(base_path + "/" + file_out + data + "_expansion_ratio.csv")
        if log_y:
            ax.set_yscale("symlog", basey=2)
         
        locations = [-1.5, -0.5, 0.5, 1.5]
        x = np.arange(len(variables))
        width = 0.2
        rects = []
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
        
        i = 0
        for (col, c) in zip(df_ratio.columns, colors):
            rects.append(ax.bar(x + locations[i] * width, df_ratio[col], width, label=algorithms[algs[i]], color=c))
            i += 1
        ax.legend(ncol=4, fontsize=12)
        fig.savefig(base_path + "/" + file_out + data + "_expansion_ratio." + file_type)

# run_memory_factors_dense_feats_dims(file_type="png")
# run_memory_factors_dense_feats_dims(file_type="pdf")
# run_inference_full_memory_feats_dims(file_type="png")
# run_inference_full_memory_feats_dims(file_type="pdf")