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