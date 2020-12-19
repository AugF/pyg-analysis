# coding=utf-8
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algorithms, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def pics_memory(file_type="png"):
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp2_time_break/dir_config_json"
    dir_out = "paper_exp3_memory"
    time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Loss', 'Backward',
                   'Eval\nLayer0', 'Eval\nLayer1']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo']
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
                all_data = all_data.T  # 
                allocated_max[algorithms[alg]] = all_data[1] # current

        allocated_max = pd.DataFrame(allocated_max, index=time_labels)

        #ax = plt.subplot(2, 3, i + 1)
        fig, ax = plt.subplots()
        ax.set_ylabel("Peak Memory Usage (MB)")
        ax.set_xlabel("Phase")
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        ax.set_xticks(list(range(len(time_labels))))
        ax.set_xticklabels(time_labels)
        for j, c in enumerate(allocated_max.columns):
            allocated_max[c].plot(ax=ax, marker=markers[j], linestyle=lines[j], label=c, rot=45)
        ax.legend()
        fig.tight_layout() 
        fig.savefig(dir_out + '/exp_memory_usage_stage_' + datasets_maps[data] + '.' + file_type)
        plt.close()


def pics_inference_full_memory(file_type="png"):
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp8_inference_full/dir_config_json"
    dir_out = "paper_exp5_inference_full"
    time_labels = ['Data\nLoad', 'Eval\nStart', 'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    for i, data in enumerate(datasets):
        allocated_max = {}
        for alg in algs:
            file_path = dir_name + '/config0_' + alg + '_' + data + '.json'
            print(file_path)
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                res = json.load(f) # res: {key, [[1, 2, 3, 4], [a, b, c, d]]}
                dataload_end = np.array(res['data load'][0][1])
                eval_start = np.array(res['eval_start']).mean(axis=0)[1]
                layer0_eval = np.array(res['layer0']).mean(axis=0)[1]
                layer1_eval = np.array(res['layer1']).mean(axis=0)[1]
                eval_end = np.array(res['eval_end']).mean(axis=0)[1]
                
                all_data = np.array([dataload_end, eval_start, layer0_eval, layer1_eval, eval_end])
                all_data /= 1024 * 1024 # GB
                allocated_max[algorithms[alg]] = all_data

        allocated_max = pd.DataFrame(allocated_max, index=time_labels)

        fig, ax = plt.subplots()
        ax.set_ylabel("Peak Memory Usage (MB)")
        ax.set_xlabel("Phase")
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        ax.set_xticks(list(range(len(time_labels))))
        ax.set_xticklabels(time_labels)
        for j, c in enumerate(allocated_max.columns):
            allocated_max[c].plot(ax=ax, marker=markers[j], linestyle=lines[j], label=c, rot=45)
        ax.legend()
        fig.tight_layout() 
        fig.savefig(dir_out + '/exp_inference_full_memory_usage_stage_' + datasets_maps[data] + '.' + file_type)
        plt.close()


pics_inference_full_memory(file_type="png")
# pics_inference_full_memory(file_type="pdf")