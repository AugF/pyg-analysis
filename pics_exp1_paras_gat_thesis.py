import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, autolabel, datasets_maps, get_inference_expansion_memory
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
xlabel = [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"]
             
def pic_calculations_gat(file_type="png", infer_flag=False,
                         file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
                         dir_in = "paper_exp1_super_parameters",
                         dir_cal = '/hidden_dims_exp/',
                         dir_out = "exp3_thesis_figs/paras",
                         ):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    labels = ['点计算', '边计算']
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    dir_subs = ["hds_head_dims_exp", "hds_heads_exp"]
    
    for alg in algs:
        for k, kx in enumerate(['head_dims', 'heads']): # 表示两种情况
            dir_path = dir_in + dir_cal + dir_subs[k] + '/calculations'
            df = {}
            df[0] = {}
            df[1] = {}
            for data in datasets: 
                file_path = dir_path + '/' + alg + '_' + data + '.csv' # 这里只与alg和data相关
                if not os.path.exists(file_path):
                    continue
                df_t = pd.read_csv(file_path, index_col=0).T
                if df_t.empty:
                    df[0][data] = [np.nan] * len(xticklabels[k])
                    df[1][data] = [np.nan] * len(xticklabels[k])
                else:
                    df[0][data] = df_t[0].values.tolist() + [np.nan] * (len(xticklabels[k]) - len(df_t[0].values))
                    df[1][data] = df_t[1].values.tolist() + [np.nan] * (len(xticklabels[k]) - len(df_t[1].values))

            df[0] = pd.DataFrame(df[0])
            df[1] = pd.DataFrame(df[1])
            for i,x in enumerate(['vertex', 'edge']):
                fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
                ax.set_title(labels[i], fontsize=base_size + 2)
                ax.set_yscale("symlog", basey=2)
                if infer_flag:
                    ax.set_ylabel("每轮推理时间 (ms)", fontsize=base_size+2)
                else:
                    ax.set_ylabel("每轮训练时间 (ms)", fontsize=base_size+2)
                ax.set_xlabel(xlabel[k], fontsize=base_size+2)
                ax.set_xticks(list(range(len(xticklabels[k]))))
                ax.set_xticklabels(xticklabels[k], fontsize=base_size)
                markers = 'oD^sdp'
                for j, c in enumerate(df[i].columns):
                    ax.plot(df[i].index, df[i][c], marker=markers[j], markersize=7, label=datasets_maps[c])
                ax.legend(ncol=2)
                fig.savefig(dir_out + "/" + file_prefix + alg + "_" + kx + "_" + x + "." + file_type, dpi=400)
        plt.close()


def run_memory_gat(file_type, infer_flag=False, 
                   file_out="exp_hyperparameter_on_memory_usage_", 
                   dir_out="exp3_thesis_figs/paras", dir_memory=""):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    file_prefix = ['_4_', '_']
    file_suffix = ['', '_32']
    
    for alg in algs:
        for k, kx in enumerate(['head_dims', 'heads']):
            df = pd.read_csv("paper_exp1_super_parameters/memory/" + file_out + alg + "_" + kx + ".csv", index_col=0)
            fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
            ax.set_yscale("symlog", basey=2)
            if infer_flag:
                ax.set_ylabel("推理阶段内存使用 (MB)", fontsize=base_size+2)
            else:
                ax.set_ylabel("训练阶段内存使用 (MB)", fontsize=base_size+2)
            ax.set_xlabel(xlabel[k], fontsize=base_size+2)
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels([str(i) for i in xticklabels[k]])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                ax.plot(df.index, df[c], marker=markers[i], markersize=7, label=datasets_maps[c])
            ax.legend()
            fig.savefig(dir_out + "/" + file_out + alg + "_" + kx + "." + file_type, dpi=400)
        plt.close()
        

if __name__ == "__main__":
    pic_calculations_gat(file_type="png")
    run_memory_gat(file_type="png",
                   dir_memory="/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gat_json")
    run_memory_gat("png", infer_flag=True, file_out="exp_hyperparameter_on_inference_memory_usage_",
                   dir_memory="/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gat_inference_json")   
    
