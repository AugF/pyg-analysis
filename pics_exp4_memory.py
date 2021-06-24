# coding=utf-8
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from utils import algorithms, datasets_maps, datasets, autolabel
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12


def pics_minibatch_memory_bar(file_type="png"):
    file_out = "exp_sampling_memory_usage_relative_batch_size_"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']

    dir_path = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp4_relative_sampling/batch_memory/"
    dir_out = "/mnt/data/wangzhaokang/wangyunpan/pyg-analysis/paper_exp4_relative_sampling"
    xlabel = "Relative Batch Size (%)"

    cluster_batchs = [15, 45, 90, 150, 375, 750]

    graphsage_batchs = {
        'amazon-photo': [77, 230, 459, 765, 1913, 3825],
        'pubmed': [198, 592, 1184, 1972, 4930, 9859],
        'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
        'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
        'flickr': [893, 2678, 5355, 8925, 22313, 44625],
        'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
    }

    xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

    log_y = True
    algs = ['gcn', 'ggnn', 'gat', 'gaan']

    for mode in ['cluster', 'graphsage']:
        for data in ["amazon-computers", "flickr"]:
            df_peak = {}
            df_ratio = {}
            for alg in algs:
                df_peak[alg] = []
                data_memory = 1e-18
                for k, var in enumerate(xticklabels):
                    if var == 'FULL':
                        file_path = dir_path + "config0_" + alg + "_" + data + "_full.json"
                        if not os.path.exists(file_path):
                            df_peak[alg].append(np.nan)
                            continue
                        with open(file_path) as f:
                            print(file_path)
                            res = json.load(f)
                            dataload_end = np.array(res['forward_start'][0])
                            warmup_end = np.array(
                                res['forward_start'][1:]).mean(axis=0)
                            layer0_forward = np.array(
                                res['layer0'][1::2]).mean(axis=0)
                            layer0_eval = np.array(
                                res['layer0'][2::2]).mean(axis=0)
                            layer1_forward = np.array(
                                res['layer1'][1::2]).mean(axis=0)
                            layer1_eval = np.array(
                                res['layer1'][2::2]).mean(axis=0)
                            forward_end = np.array(
                                res['forward_end'][1:]).mean(axis=0)
                            backward_end = np.array(
                                res['backward_end'][1:]).mean(axis=0)
                            eval_end = np.array(res['eval_end']).mean(axis=0)
                            all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                                 backward_end, layer0_eval, layer1_eval, eval_end])

                            all_data /= (1024 * 1024)
                            df_peak[alg].append(max(all_data[2:, 1]))
                            data_memory = all_data[0][0]
                    else:
                        if mode == 'cluster':
                            file_path = dir_path + mode + '_' + alg + '_' + \
                                data + '_' + str(cluster_batchs[k]) + ".json"
                        else:
                            file_path = dir_path + mode + '_' + alg + '_' + data + \
                                '_' + str(graphsage_batchs[data][k]) + ".json"
                        if not os.path.exists(file_path):
                            df_peak[alg].append(np.nan)
                            continue
                        with open(file_path) as f:
                            res = json.load(f)
                            print(file_path)
                            model_load = np.array(res['model load'][0])
                            warmup_end = np.array(res['warmup end'][0])
                            batch_start = np.array(
                                res['batch_start'][1:]).mean(axis=0)
                            layer0 = np.array(res['layer0'][1:]).mean(
                                axis=0)  # 去除warmup后的最大值
                            layer1 = np.array(res['layer1'][1:]).mean(axis=0)
                            to_end = np.array(res['to end'][1:]).mean(axis=0)
                            forward_end = np.array(
                                res['forward_end'][1:]).mean(axis=0)
                            backward_end = np.array(
                                res['backward_end'][1:]).mean(axis=0)
                            all_data = np.array(
                                [model_load, warmup_end, batch_start, layer0, layer1, to_end, forward_end, backward_end])

                            all_data /= (1024 * 1024)
                            df_peak[alg].append(max(all_data[2:, 1]))

                print(data_memory)
                df_ratio[alg] = [x / data_memory for x in df_peak[alg]]

            # 得到有效index, 去除无效index
            enabels_indexs = []
            labels = []
            for i, var in enumerate(xticklabels):
                flag = False
                for alg in algs:
                    if str(df_peak[alg][i]) != 'nan' and str(df_ratio[alg][i]) != 'nan':
                        flag = True
                        break
                if flag:
                    enabels_indexs.append(i)
                    labels.append(var)

            # 指定bar的location
            locations = [-1.5, -0.5, 0.5, 1.5]
            x = np.arange(len(labels))
            width = 0.2
            colors = plt.get_cmap('Paired')(
                np.linspace(0.15, 0.85, len(locations)))

            fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
            ax.set_ylabel("Peak Memory Usage (MB)", fontsize=18)
            ax.set_xlabel(xlabel, fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # if log_y:
            #     ax.set_yscale("symlog", basey=2)

            for i, alg in enumerate(algs):
                ax.bar(x + locations[i] * width, [df_peak[alg][d]
                                                  for d in enabels_indexs], width, label=algorithms[alg], color=colors[i])
            ax.set_xticklabels([0] + labels)
            ax.legend(fontsize=16)
            fig.savefig(dir_out + "/" + file_out + mode + '_' +
                        data + "_peak_memory." + file_type)

            # fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
            # ax.set_ylabel("Expansion Ratio", fontsize=16)
            # # if log_y:
            # #     ax.set_yscale("symlog", basey=2)
            # ax.set_xlabel(xlabel, fontsize=16)
            # plt.xticks(fontsize=14)
            # plt.yticks(fontsize=14)
            # for i, alg in enumerate(algs):
            #     ax.bar(x + locations[i] * width, [df_ratio[alg][d]
            #                                       for d in enabels_indexs], width, label=algorithms[alg], color=colors[i])
            # ax.set_xticklabels([0] + labels)
            # ax.legend(fontsize=12)
            # fig.savefig(dir_out + "/" + file_out + mode + '_' +
            #             data + "_expansion_ratio." + file_type)

            plt.close()


def pics_inference_sampling_memory(dir_work="inference_sampling_memory", file_suffix="", file_type="png"):
    file_out = "exp_inference_sampling_fix_batch_size_memory_usage_" + file_suffix
    dir_in = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp7_inference_sampling/" + dir_work
    dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/new_exp_supplement"
    xlabel = "Dataset"

    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers',
                'coauthor-physics', 'flickr', 'com-amazon']

    df_peak_memory, df_memory_ratio = {}, {}
    for alg in algs:
        df_peak_memory[alg], df_memory_ratio[alg] = [], []
        for data in datasets:
            file_path = dir_in + '/' + alg + '_' + data + '.json'
            print(file_path)
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                res = json.load(f)
                data_memory = res['data load'][0][0]
                max_memory = 0
                for k in res.keys():
                    max_memory = max(max_memory, np.array(
                        res[k]).mean(axis=0)[1])
                df_memory_ratio[alg].append(max_memory / data_memory)
                max_memory /= 1024 * 1024 * 1024  # GB
                df_peak_memory[alg].append(max_memory)

    # 指定bar的location
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(datasets))
    width = 0.2
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))

    # fig1: inference sampling阶段的peak memory的图像
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel("Peak Memory Usage (MB)", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylim(0, 2.5)

    for i, alg in enumerate(algs):
        ax.bar(x + locations[i] * width, df_peak_memory[alg],
               width, label=algorithms[alg], color=colors[i])
    ax.set_xticklabels([''] + [datasets_maps[x] for x in datasets])
    ax.legend(fontsize=12)
    fig.savefig(dir_out + "/" + file_out + "_peak_memory." + file_type)

    # fig2: inference sampling阶段的memory ratio的图像
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.set_ylabel("Expansion Ratio", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylim(0, 11)

    for i, alg in enumerate(algs):
        ax.bar(x + locations[i] * width, df_memory_ratio[alg],
               width, label=algorithms[alg], color=colors[i])
    ax.set_xticklabels([''] + [datasets_maps[x] for x in datasets])
    ax.legend(fontsize=12)
    fig.savefig(dir_out + "/" + file_out + "_expansion_ratio." + file_type)

    plt.close()


pics_minibatch_memory_bar(file_type="png")
pics_minibatch_memory_bar(file_type="pdf")
# pics_inference_sampling_memory(
#     dir_work="inference_sampling_memory_2048", file_suffix="2048", file_type="png")
# pics_inference_sampling_memory(
#     dir_work="inference_sampling_memory_2048", file_suffix="2048", file_type="pdf")
# pics_inference_sampling_memory(
#     dir_work="inference_sampling_memory", file_suffix="1024", file_type="png")
# pics_inference_sampling_memory(
#     dir_work="inference_sampling_memory", file_suffix="1024", file_type="pdf")
