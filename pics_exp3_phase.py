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
    dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/new_exp_supplement"
    # time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Loss', 'Backward',
    #                'Eval\nLayer0', 'Eval\nLayer1']
    time_labels = ['数据加载', 'GPU预热', '前向传播Layer0', '前向传播Layer1', '损失计算', '后向传播', '评估Layer0', '评估Layer1']
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
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_ylabel("Peak Memory Usage (MB)", fontsize=16)
        ax.set_xlabel("Phase", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_ylim(0, 2500)
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        ax.set_xticks(list(range(len(time_labels))))
        ax.set_xticklabels(time_labels)
        for j, c in enumerate(allocated_max.columns):
            allocated_max[c].plot(ax=ax, marker=markers[j], linestyle=lines[j], label=c, rot=45)
        ax.legend(fontsize=12)
        fig.tight_layout() 
        fig.savefig(dir_out + '/exp_memory_usage_stage_' + datasets_maps[data] + '.' + file_type)
        plt.close()


def pics_inference_full_memory(file_type="png"):
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp8_inference_full/dir_config_json"
    dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/new_exp_supplement"
    # time_labels = ['Data\nLoad', 'Eval\nStart', 'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']
    time_labels = ['数据加载', '推理开始', '推理Layer0', '推理Layer1', '推理结束']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    for i, data in enumerate(["amazon-photo"]):
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

        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        ax.set_ylabel("Peak Memory Usage (MB)", fontsize=16)
        ax.set_xlabel("Phase", fontsize=16)
        ax.set_ylim(0, 2500)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        ax.set_xticks(list(range(len(time_labels))))
        ax.set_xticklabels(time_labels)
        for j, c in enumerate(allocated_max.columns):
            allocated_max[c].plot(ax=ax, marker=markers[j], linestyle=lines[j], label=c, rot=45)
        ax.legend(fontsize=12)
        fig.tight_layout() 
        fig.savefig(dir_out + '/exp_inference_full_memory_usage_stage_' + datasets_maps[data] + '.' + file_type)
        plt.close()

pics_memory(file_type="png")
pics_memory(file_type="pdf")
pics_inference_full_memory(file_type="png")
pics_inference_full_memory(file_type="pdf")
