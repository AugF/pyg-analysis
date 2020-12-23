import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps, autolabel
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def run_memory_ratio_config_single(file_type="png"):
    file_out="exp_memory_expansion_ratio"
    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp2_time_break/dir_config_json"
    dir_out = "paper_exp3_memory"
    
    df = []
    for alg in algs:
        results = []
        for data in datasets:
            file_path = dir_memory + '/config0_' + alg + '_' + data + '.json'
            if not os.path.exists(file_path):
                results.append(np.nan)
                continue
            with open(file_path) as f:
                res = json.load(f)
                dataload_end = np.array(res['forward_start'][0])
                print(dataload_end)
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
                print(max(all_data[2:, 1]), "data", all_data[0, 0])
                results.append(max(all_data[2:, 1]) / all_data[0, 0]) # max memory / data loader current 
        df.append(results.copy())
    
    labels = [datasets_maps[d] for d in datasets]
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(labels))
    width = 0.2
    rects = []
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
    fig, ax = plt.subplots()
    i = 0
    for (l, c) in zip(locations, colors):
        rects.append(ax.bar(x + l * width, df[i], width, label=algorithms[algs[i]], color=c))
        i += 1
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for r in rects:
        ax = autolabel(r, ax, memory_ratio_flag=True)
    ax.legend(ncol=2)
    fig.savefig(dir_out + "/" + file_out +  "." + file_type)
    plt.close()


def run_inference_full_memory_ratio(file_type="png"):
    file_out="exp_inference_full_memory_expansion_ratio"
    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp8_inference_full/dir_config_json"
    dir_out = "paper_exp5_inference_full"
    
    # 不同的算法，不同的数据集画不同的图
    df = []
    for alg in algs:
        results = []
        for data in datasets:
            file_path = dir_memory + '/config0_' + alg + '_' + data + '.json'
            if not os.path.exists(file_path):
                results.append(np.nan)
                continue
            with open(file_path) as f:
                res = json.load(f)
                data_memory = res['data load'][0][0]
                max_memory = 0
                for k in res.keys():
                    max_memory = max(max_memory, np.array(res[k]).mean(axis=0)[1])
                
                print("data memory", data_memory)
                results.append(max_memory / data_memory) # max memory / data loader current 
        df.append(results.copy())
    
    labels = [datasets_maps[d] for d in datasets]
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(labels))
    width = 0.2
    rects = []
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
    fig, ax = plt.subplots()
    i = 0
    for (l, c) in zip(locations, colors):
        rects.append(ax.bar(x + l * width, df[i], width, label=algorithms[algs[i]], color=c))
        i += 1
    ax.set_ylabel("Expansion Ratio")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for r in rects:
        ax = autolabel(r, ax, memory_ratio_flag=True)
    ax.legend(ncol=2)
    fig.savefig(dir_out + "/" + file_out +  "." + file_type)
    plt.close()
    

run_memory_ratio_config_single(file_type="png")
run_memory_ratio_config_single(file_type="pdf")
run_inference_full_memory_ratio(file_type="png")
run_inference_full_memory_ratio(file_type="pdf")