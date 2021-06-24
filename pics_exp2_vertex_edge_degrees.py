import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps
from matplotlib.font_manager import _rebuild
_rebuild() 
base_size = 10
plt.rcParams["font.size"] = base_size

def pic_calculations_avg_degree(dir_out="paper_exp2_time_break", dir_work="paper_exp2_time_break",
                                file_prefix = "exp_avg_degree_on_vertex_edge_cal_time_", file_type="png"):
    xticklabels = [3, 6, 10, 15, 20, 25, 30, 50]
    xlabel = "平均顶点度数"
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    labels = ['点计算', '边计算']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    
    dir_path = dir_work + '/degrees_exp/calculations'
    for alg in algs:
        fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)
        df = pd.read_csv(dir_path + '/' + alg + '_graph.csv', index_col=0).T
        
        ax.set_xlabel(xlabel, fontsize=base_size+2)
        if 'inference' in dir_work:
            ax.set_ylabel("平均每轮推理时间 (毫秒)", fontsize=base_size+2)
        else:
            ax.set_ylabel("平均每轮训练时间 (毫秒)", fontsize=base_size+2)
        xticks = xticklabels[:len(df.index)]
        line1, = ax.plot(xticks, df[0].values, 'ob', label=labels[0], linestyle='-')
        line2, = ax.plot(xticks, df[1].values, 'Dg', label=labels[1], linestyle='-')
        
        edge_percent = df[1].values * 100 / df.values.sum(axis=1)
        ax2 = ax.twinx()
        ax2.set_ylabel("边计算耗时占比 (%)", fontsize=base_size + 2)
        line3, = ax2.plot(xticks, edge_percent, 'rs--', label='边计算比例')
        plt.legend(handles=[line1, line2, line3], loc="upper left", fontsize=base_size)
        plt.xticks(fontsize=base_size)
        plt.yticks(fontsize=base_size)
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + "." + file_type)
        plt.close()


pic_calculations_avg_degree(dir_out="paper_exp2_time_break/paper_figs", dir_work="paper_exp2_time_break",
                            file_prefix = "exp_avg_degree_on_vertex_edge_cal_time_", file_type="png")
# pic_calculations_avg_degree(dir_out="exp_supplement", dir_work="paper_exp2_time_break",
#                             file_prefix = "exp_avg_degree_on_vertex_edge_cal_time_", file_type="pdf")
# pic_calculations_avg_degree(dir_out="exp_supplement", dir_work="paper_exp5_inference_full",
#                             file_prefix = "exp_inference_full_avg_degree_on_vertex_edge_cal_time_", file_type="png")
# pic_calculations_avg_degree(dir_out="exp_supplement", dir_work="paper_exp5_inference_full",
#                             file_prefix = "exp_inference_full_avg_degree_on_vertex_edge_cal_time_", file_type="pdf")