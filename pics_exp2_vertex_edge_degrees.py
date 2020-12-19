import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def pic_calculations_avg_degree(dir_work="paper_exp2_time_break", file_prefix = "exp_avg_degree_on_vertex_edge_cal_time_", file_type="png"):
    xticklabels = [3, 6, 10, 15, 20, 25, 30, 50]
    xlabel = "Average Vertex Degree"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    labels = ['Vertex Calculation', 'Edge Calculation']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    
    dir_path = dir_work + '/degrees_exp/calculations'
    for alg in algs:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        df = pd.read_csv(dir_path + '/' + alg + '_graph.csv', index_col=0).T
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Training Time / Epoch (ms)")
        xticks = xticklabels[:len(df.index)]
        line1, = ax.plot(xticks, df[0].values, 'ob', label=labels[0], linestyle='-')
        line2, = ax.plot(xticks, df[1].values, 'Dg', label=labels[1], linestyle='-')
        
        edge_percent = df[1].values * 100 / df.values.sum(axis=1)
        ax2 = ax.twinx()
        ax2.set_ylabel("Proportion of Edge Calculation(%)")
        line3, = ax2.plot(xticks, edge_percent, 'rs--', label='Edge Calculation Proportion')
        plt.legend(handles=[line1, line2, line3], loc="upper left")
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_work + "/" + file_prefix + alg + "." + file_type)
        plt.close()

pic_calculations_avg_degree(dir_work="paper_exp5_inference_full", file_prefix = "exp_inference_full_avg_degree_on_vertex_edge_cal_time_", file_type="png")
# pic_calculations_avg_degree(file_type="pdf")