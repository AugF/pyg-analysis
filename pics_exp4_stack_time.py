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

def pics_minibatch_time(file_type="png"):
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp4_relative_sampling/batch_train_time_stack"
    dir_out = "paper_exp4_relative_sampling"
    file_out = "exp_sampling_relative_batch_size_train_time_stack"
    
    ylabel = "Training Time per Batch (ms)"
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

    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    
    for mode in ['cluster', 'graphsage']:
        # for data in ["amazon-computers", "flickr"]:
        for data in graphsage_batchs.keys():
            df_train = {}
            df_sampling = {}
            df_to = {}
            for alg in algs:
                df_train[alg] = []
                df_sampling[alg] = []
                df_to[alg] = []
                for k, var in enumerate(xticklabels):
                    if var == 'FULL':
                        file_path = dir_path + "/config0_" + alg + "_" + data + "_full.log"
                    else:
                        if mode == 'cluster':
                            file_path = dir_path + '/' + mode + '_' + alg + '_' + data + '_' + str(cluster_batchs[k]) + ".log"
                        else:
                            file_path = dir_path + '/' + mode + '_' + alg + '_' + data + '_' + str(graphsage_batchs[data][k]) + ".log"
                    
                    if not os.path.exists(file_path):
                        df_sampling[alg].append(np.nan)
                        df_to[alg].append(np.nan)
                        df_train[alg].append(np.nan)
                        continue
                    print(file_path)
                    train_time = 0.0
                    sampling_time, to_time = 0.0, 0.0
                    with open(file_path) as f:
                        for line in f:
                            match_line = re.match(r".*, avg_batch_train_time: (.*), avg_batch_sampling_time:(.*), avg_batch_to_time: (.*)", line)
                            match_line2 = re.match(r"train_time_per_epoch:  (.*)", line)
                            if match_line:
                                train_time = float(match_line.group(1))
                                sampling_time = float(match_line.group(2))
                                to_time = float(match_line.group(3))
                                print(var, train_time, sampling_time, to_time)
                                break
                            if match_line2:
                                train_time = float(match_line2.group(1))
                                print(var, train_time)
                                break
                    if train_time == 0.0:
                        df_train[alg].append(np.nan)
                        df_sampling[alg].append(np.nan)
                        df_to[alg].append(np.nan)
                    else:
                        df_train[alg].append(train_time * 1000)
                        df_to[alg].append(to_time * 1000)
                        df_sampling[alg].append(sampling_time * 1000)

            fig, ax = plt.subplots()
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            enabels_indexs = [] 
            labels = []           
            for i, var in enumerate(xticklabels):
                flag = False
                for alg in algs:
                    if str(df_train[alg][i]) != 'nan' or str(df_to[alg][i]) != 'nan' or str(df_sampling[alg][i]) != 'nan':
                        flag = True
                        break
                if flag:
                    enabels_indexs.append(i)
                    labels.append(var)
            
            ax.set_xticklabels([0] + labels)
            file_name = mode + '_' + data
            # ylim = 0
            # if data == 'pubmed' or file_name in ['cluster_amazon-computers', 'cluster_amazon-photo']:
            #     ylim = 60
            # elif mode == 'cluster':
            #     ylim = 100
            # elif file_name != 'graphsage_coauthor-physics':
            #     ylim = 130
            # else:
            #     ylim = 700
            # ax.set_ylim(0, ylim)
            
            out_path = dir_out + '/batch_train_time_stack/' + mode + '_' + data
            pd.DataFrame(df_train).to_csv(out_path + "_train_time.csv")
            pd.DataFrame(df_to).to_csv(out_path + "_to_time.csv")
            pd.DataFrame(df_sampling).to_csv(out_path + "_sampling_time.csv")
            
            locations = [-1.5, -0.5, 0.5, 1.5]
            x = np.arange(len(labels))
            width = 0.2
            colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))

            rects = []
            i = 0
            for alg in algs:
                tmp_train = [df_train[alg][j] for j in enabels_indexs]
                tmp_to = [df_to[alg][j] for j in enabels_indexs]
                tmp_sampling = [df_sampling[alg][j] for j in enabels_indexs]
                # print("train", tmp_train)
                # print("to", tmp_to)
                # print("sampling", tmp_sampling)
                # print(x + locations[i] * width)
                rects.append(ax.bar(x + locations[i] * width, tmp_train, width, color=colors[i], edgecolor='black', hatch="////"))
                
                rects.append(ax.bar(x + locations[i] * width, tmp_to, width, color=colors[i], edgecolor='black', bottom=tmp_train, hatch='....'))
                tmp = [tmp_train[j] + tmp_to[j] for j in range(len(tmp_train))]
                # print("tmp", tmp)
                rects.append(ax.bar(x + locations[i] * width, tmp_sampling, width, color=colors[i], edgecolor='black', bottom=tmp, hatch='xxxx'))
                i += 1
            
            legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
            legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='////'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='xxxx')]
            ax.legend(legend_colors + legend_hatchs, [algorithms[i] for i in algs] + ['Training', 'Data Transferring', 'Sampling'], ncol=2)

            # ax.bar([-1], [-1], hatch='///', label='Training')

            # for rect in rects:
            #     for r in rect:
            #         height = r.get_height()
            #         if str(height) == 'nan':
            #             continue
            #         if height >= ylim:
            #             ax.text(r.get_x() + 0.2, ylim - 5, int(height), fontsize=7.5)
            
            print(dir_out + '/' + file_out + mode + "_" + data + "." + file_type)
            fig.savefig(dir_out + '/' + file_out + mode + "_" + data + "." + file_type)
            plt.close()


def pics_inference_sampling_minibatch_time(file_type="png"):
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp7_inference_sampling/inference_sampling_time"
    dir_out = "paper_exp6_inference_sampling"
    file_out = "exp_inference_sampling_relative_batch_size_train_time_stack"
    
    ylabel = "Training Time per Batch (ms)"
    xlabel = "Datasets"
    
    xticklabels = [datasets_maps[i] for i in datasets]

    algs = ['gcn', 'ggnn', 'gat', 'gaan']

    df_train = {}
    df_sampling = {}
    df_to = {}
    for alg in algs:
        df_train[alg] = []
        df_sampling[alg] = []
        df_to[alg] = []
        for data in datasets:
            file_path = dir_path + '/' + alg + '_' + data + ".log"
            if not os.path.exists(file_path):
                df_sampling[alg].append(np.nan)
                df_to[alg].append(np.nan)
                df_train[alg].append(np.nan)
                continue
            # print(file_path)
            sampling_time, to_time, train_time = 0.0, 0.0, 0.0
            with open(file_path) as f:
                for line in f:
                    match_line = re.match(r"avg_batch_train_time: (.*), avg_batch_sampling_time:(.*), avg_batch_to_time: (.*)", line)
                    if match_line:
                        train_time += float(match_line.group(1))
                        sampling_time += float(match_line.group(2))
                        to_time += float(match_line.group(3))
            if train_time == 0.0:
                df_train[alg].append(np.nan)
                df_sampling[alg].append(np.nan)
                df_to[alg].append(np.nan)
            else:
                df_train[alg].append(train_time * 1000)
                df_to[alg].append(to_time * 1000)
                df_sampling[alg].append(sampling_time * 1000)

    # fig: 画inference sampling的图像
    fig, ax = plt.subplots()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels([''] + xticklabels)
        
    pd.DataFrame(df_train).to_csv(dir_out + "/" + file_out + "_train_time.csv")
    pd.DataFrame(df_to).to_csv(dir_out + "/" + file_out + "_to_time.csv")
    pd.DataFrame(df_sampling).to_csv(dir_out + "/" + file_out + "_sampling_time.csv")
    
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(datasets))
    width = 0.2
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))

    rects = []
    i = 0
    for alg in algs:
        tmp_train, tmp_to, tmp_sampling = df_train[alg], df_to[alg], df_sampling[alg]
        rects.append(ax.bar(x + locations[i] * width, tmp_train, width, color=colors[i], edgecolor='black', hatch="////"))
        rects.append(ax.bar(x + locations[i] * width, tmp_to, width, color=colors[i], edgecolor='black', bottom=tmp_train, hatch='....'))
        tmp = [tmp_train[j] + tmp_to[j] for j in range(len(tmp_train))]
        rects.append(ax.bar(x + locations[i] * width, tmp_sampling, width, color=colors[i], edgecolor='black', bottom=tmp, hatch='xxxx'))
        i += 1

    legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
    legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='////'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='xxxx')]
    ax.legend(legend_colors + legend_hatchs, [algorithms[i] for i in algs] + ['Training', 'Data Transferring', 'Sampling'], ncol=2)
    
    fig.savefig(dir_out + '/' + file_out +  "." + file_type)
    plt.close()


pics_inference_sampling_minibatch_time(file_type="png")