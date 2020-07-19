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

# sampling
def pics_stages_bar(file_type="png"):
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/stages_time_log/"
    modes = ['cluster', 'graphsage']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = 'sampling_relative_exp/'
    file_out = 'exp_sampling_time_decomposition_'
    xticklabels = ['Sampling', 'Data Transferring', 'Training']
    
    for mode in modes:
        df_mean = {}
        df_min = {}
        df_max = {}
        for alg in algs:
            df_mean[alg] = [0.0] * 3
            df_min[alg] = [sys.maxsize] * 3
            df_max[alg] = [0.0] * 3
            cnt = 0
            for data in datasets:
                file_path = dir_path + alg + '_' + data + '_' + mode + '.log'
                if not os.path.exists(file_path):
                    continue
                print(file_path)
                sampling_time, to_time, train_time = 0, 0, 0
                with open(file_path) as f:
                    for line in f:
                        matchs_line = re.match(r".*, avg_epoch_train_time: (.*), avg_epochs_sampling_time:(.*), avg_epoch_to_time: (.*)", line)
                        if matchs_line:
                            sampling_time, to_time, train_time = matchs_line.group(2), matchs_line.group(3), matchs_line.group(1)
                            break
                #print(sampling_time, to_time, train_time)
                if sampling_time == 0 and to_time == 0 and train_time == 0:
                    continue
                cnt += 1
                all_time = [float(x) for x in [sampling_time, to_time, train_time]]
                sum_time = sum(all_time)
                all_time = [100 * x / sum_time for x in all_time]
                
                for j, x in enumerate(all_time):
                    df_mean[alg][j] += x
                    df_min[alg][j] = min(df_min[alg][j], x)
                    df_max[alg][j] = max(df_max[alg][j], x)

            for j in range(3):
                df_mean[alg][j] /= cnt
                df_min[alg][j] = df_mean[alg][j] - df_min[alg][j]
                df_max[alg][j] -= df_mean[alg][j]
             
        locations = [-1.5, -0.5, 0.5, 1.5]
        x = np.arange(len(xticklabels))
        width = 0.2
        rects = []
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
        fig, ax = plt.subplots()
        i = 0
        for (l, c) in zip(locations, colors):
            print(algs[i], df_mean[algs[i]], df_min[algs[i]], df_max[algs[i]])
            rects.append(ax.bar(x + l * width, df_mean[algs[i]], width, label=algorithms[algs[i]], color=c, yerr=[df_min[algs[i]], df_max[algs[i]]]))
            i += 1
        ax.set_ylabel("Proportion (%)")
        ax.set_xlabel("Stages")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.legend()
        fig.savefig(dir_out + file_out + mode + '.' + file_type)                        


def pics_minbatch_graph_size(file_type="png"):
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/batch_graph_info/"
    dir_out = "sampling_relative_exp/"
    file_out = "exp_sampling_minibatch_realtive_graph_info_"
    
    modes = ['cluster', 'graphsage']
    algs = ['gcn']
    # variables = {
    #         'cluster': [2, 4, 8, 16, 32, 64, 128],
    #         'graphsage': [256, 512, 1024, 2048, 4096]
    #         }
    cluster_batchs = [15, 45, 90, 150, 375, 750]

    graphsage_batchs = {
        'amazon-photo': [77, 230, 459, 765, 1913, 3825],
        'pubmed': [198, 592, 1184, 1972, 4930, 9859],
        'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
        'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
        'flickr': [893, 2678, 5355, 8925, 22313, 44625],
        'com-amazon': [3349, 10046, 20092, 33487, 83716, 167432]
    }
    
    xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%']
    variabels = [1, 3, 6, 10, 25, 50]
    ylabels = ["Number of Vertices", "Number of Edges", "Average Degree"]
    xlabel = 'Relative Batch Size (%)'
    markers = 'oD^sdp'

    for mode in modes:
        for alg in algs:
            plt.figure(figsize=(18, 4.8))
            fig = plt.figure(1)
            df_edges = {}
            df_degrees = {}
            df_nodes = {}
            for data in datasets:
                df_edges[data] = {}
                df_edges[data]['mean'] = []
                df_edges[data]['std'] = []
                df_degrees[data] = {}
                df_degrees[data]['mean'] = []
                df_degrees[data]['std'] = []
                df_nodes[data] = {}
                df_nodes[data]['mean'] = []
                df_nodes[data]['std'] = []
                for k, var in enumerate(xticklabels):
                    if mode == 'cluster':
                        file_path = dir_path + mode + '_' + alg + '_' + data + '_' + str(cluster_batchs[k]) + ".log"
                    else:
                        file_path = dir_path + mode + '_' + alg + '_' + data + '_' + str(graphsage_batchs[data][k]) + ".log"
                    print(file_path)
                    nodes_list, edges_list, degrees_list = [], [], []
                    with open(file_path) as f:
                        for line in f:
                             match_lines = re.match(r"nodes: (.*), edges: (.*)", line)
                             if match_lines:
                                 nodes, edges = int(match_lines.group(1)), int(match_lines.group(2))
                                 nodes_list.append(nodes)
                                 edges_list.append(edges)
                                 degrees_list.append(edges * 1.0 / nodes)
                    print(mode, alg, data, var, np.mean(degrees_list))
                    df_nodes[data]['mean'].append(np.mean(nodes_list))
                    df_nodes[data]['std'].append(np.std(nodes_list))
                    df_edges[data]['mean'].append(np.mean(edges_list))
                    df_edges[data]['std'].append(np.std(edges_list))
                    df_degrees[data]['mean'].append(np.mean(degrees_list))
                    df_degrees[data]['std'].append(np.std(degrees_list))
            # ax1
            ax1 = plt.subplot(1, 3, 1)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabels[0])
            
            
            for i, data in enumerate(datasets):
                ax1.errorbar(variabels, df_nodes[data]['mean'], yerr=df_nodes[data]['std'], label=datasets_maps[data], marker=markers[i])
            ax1.legend()

            # ax2
            ax2 = plt.subplot(1, 3, 2)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabels[1])
            
            for i, data in enumerate(datasets):
                ax2.errorbar(variabels, df_edges[data]['mean'], yerr=df_edges[data]['std'], label=datasets_maps[data], marker=markers[i])
            ax2.legend()

            # ax3
            ax3 = plt.subplot(1, 3, 3)
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel(ylabels[2])
            
            for i, data in enumerate(datasets):
                ax3.errorbar(variabels, df_degrees[data]['mean'], yerr=df_degrees[data]['std'], label=datasets_maps[data], marker=markers[i])
            ax3.legend()

            fig.tight_layout()
            fig.savefig(dir_out + file_out + mode + '_' + alg + "." + file_type)
            plt.close()
    

def pics_minibatch_time(file_type="png"):
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/batch_train_time_stack/"
    dir_out = "sampling_relative_exp/"
    file_out = "exp_sampling_relative_batch_size_train_time_stack_"
    
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
        for data in ["amazon-computers", "flickr"]:
            df_train = {}
            df_sampling = {}
            df_to = {}
            for alg in algs:
                df_train[alg] = []
                df_sampling[alg] = []
                df_to[alg] = []
                for k, var in enumerate(xticklabels):
                    if var == 'FULL':
                        file_path = dir_path + "config0_" + alg + "_" + data + "_full.log"
                    else:
                        if mode == 'cluster':
                            file_path = dir_path + mode + '_' + alg + '_' + data + '_' + str(cluster_batchs[k]) + ".log"
                        else:
                            file_path = dir_path + mode + '_' + alg + '_' + data + '_' + str(graphsage_batchs[data][k]) + ".log"
                    
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
                flag = True
                for alg in algs:
                    if str(df_train[alg][i]) == 'nan' or str(df_to[alg][i]) == 'nan' or str(df_sampling[alg][i]) == 'nan':
                        flag = False
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
            
            out_path = dir_out + '/batch_time/' + mode + '_' + data
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
            ax.legend(legend_colors + legend_hatchs, [algorithms[i] for i in algs] + ['Training', 'Data Transfering', 'Sampling'], ncol=2)

            # ax.bar([-1], [-1], hatch='///', label='Training')

            # for rect in rects:
            #     for r in rect:
            #         height = r.get_height()
            #         if str(height) == 'nan':
            #             continue
            #         if height >= ylim:
            #             ax.text(r.get_x() + 0.2, ylim - 5, int(height), fontsize=7.5)
            
            print(dir_out + file_out + mode + "_" + data + "." + file_type)
            fig.savefig(dir_out + file_out + mode + "_" + data + "." + file_type)
            plt.close()


# added in 7.9
def pics_minibatch_memory_bar(file_type="png"):
    file_out="exp_sampling_memory_usage_relative_batch_size_"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/batch_relative_memory/"
    base_path = "sampling_relative_exp"
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
                            df_peak[alg].append(max(all_data[2:, 1]))
                            data_memory = all_data[0][0]
                    else:
                        if mode == 'cluster':
                            file_path = dir_path + mode + '_' + alg + '_' + data + '_' + str(cluster_batchs[k]) + ".json"
                        else:
                            file_path = dir_path + mode + '_' + alg + '_' + data + '_' + str(graphsage_batchs[data][k]) + ".json"
                        if not os.path.exists(file_path):
                            df_peak[alg].append(np.nan)
                            continue
                        with open(file_path) as f:
                            res = json.load(f)
                            # print(file_path)
                            model_load = np.array(res['model load'][0])
                            warmup_end = np.array(res['warmup end'][0])
                            batch_start = np.array(res['batch_start'][1:]).mean(axis=0)
                            layer0 = np.array(res['layer0'][1:]).mean(axis=0) # 去除warmup后的最大值
                            layer1 = np.array(res['layer1'][1:]).mean(axis=0)
                            to_end = np.array(res['to end'][1:]).mean(axis=0)   
                            forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                            backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                            all_data = np.array([model_load, warmup_end, batch_start, layer0, layer1, to_end, forward_end, backward_end])
                            
                            all_data /= (1024 * 1024)
                            df_peak[alg].append(max(all_data[2:, 1]))
                            
                # print(data_memory)
                df_ratio[alg] = [x / data_memory for x in df_peak[alg]]
            
            # 得到有效index, 去除无效index
            enabels_indexs = [] 
            labels = []           
            for i, var in enumerate(xticklabels):
                flag = True
                for alg in algs:
                    if str(df_peak[alg][i]) == 'nan' or str(df_ratio[alg][i]) == 'nan':
                        flag = False
                        break
                if flag:
                    enabels_indexs.append(i)
                    labels.append(var)
            
            # 指定bar的location
            locations = [-1.5, -0.5, 0.5, 1.5]
            x = np.arange(len(labels))
            width = 0.2
            colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
            
            fig, ax = plt.subplots()
            ax.set_ylabel("Peak Memory Usage (MB)")
            ax.set_xlabel(xlabel)
            # if log_y:
            #     ax.set_yscale("symlog", basey=2)

            for i, alg in enumerate(algs):
                ax.bar(x + locations[i] * width, [df_peak[alg][d] for d in enabels_indexs], width, label=algorithms[alg], color=colors[i])
            ax.set_xticklabels([0] + labels)
            ax.legend()
            fig.savefig(base_path + "/" + file_out + mode + '_' + data + "_peak_memory." + file_type)
        
            fig, ax = plt.subplots()
            ax.set_ylabel("Expansion Ratio")
            # if log_y:
            #     ax.set_yscale("symlog", basey=2)
            ax.set_xlabel(xlabel)

            for i, alg in enumerate(algs):
                ax.bar(x + locations[i] * width, [df_ratio[alg][d] for d in enabels_indexs], width, label=algorithms[alg], color=colors[i])
            ax.set_xticklabels([0] + labels)
            ax.legend()
            fig.savefig(base_path + "/" + file_out + mode + '_' + data +  "_expansion_ratio." + file_type)
            
            plt.close()


pics_stages_bar(file_type="pdf")
# pics_minibatch_time(file_type="pdf")
# pics_minibatch_memory_bar(file_type="pdf")
# pics_minbatch_graph_size(file_type="pdf")
