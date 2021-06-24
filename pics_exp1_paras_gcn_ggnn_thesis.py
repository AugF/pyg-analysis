import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, datasets_maps, get_inference_expansion_memory
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


def pic_calculations_gcn_ggnn(file_type, infer_flag=False,
        file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
        dir_in="/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters",
        dir_cal="/hidden_dims_exp/hds_exp/calculations/",
        dir_out="exp3_thesis_figs/paras"):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    xlabel = "隐藏向量维度\n" + r"($dim(\mathbf{h}^1_x)$)"
    # for GCN and GGNN
    xticklabels = ['16', '32', '64', '128', '256', '512', '1024', '2048']
    algs = ['gcn', 'ggnn']
    labels = ['点计算', '边计算']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers',
                'coauthor-physics', 'flickr', 'com-amazon']

    for alg in algs:
        df = {}
        df[0] = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_in + dir_cal + \
                alg + '_' + data + '.csv'  # 这里只与alg和data相关

            if not os.path.exists(file_path):
                continue
            df_t = pd.read_csv(file_path, index_col=0).T
            if df_t.empty:
                df[0][data] = [np.nan] * len(xticklabels)
                df[1][data] = [np.nan] * len(xticklabels)
            else:
                df[0][data] = df_t[0].values.tolist() + [np.nan] * \
                    (len(xticklabels) - len(df_t[0].values))
                df[1][data] = df_t[1].values.tolist() + [np.nan] * \
                    (len(xticklabels) - len(df_t[1].values))

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
            ax.set_xlabel(xlabel, fontsize=base_size + 2)
            ax.set_xticks(list(range(len(xticklabels))))
            ax.set_xticklabels(xticklabels, fontsize=base_size, rotation=30)
            markers = 'oD^sdp'
            for j, c in enumerate(df[i].columns):
                ax.plot(df[i].index, df[i][c], marker=markers[j], markersize=7,
                        label=datasets_maps[c])
            ax.legend()
            fig.savefig(dir_out + "/" + file_prefix + alg + "_" + x + "." + file_type, dpi=400)
        plt.close()


def run_memory_gcn_ggnn(file_type, infer_flag=False,
        file_out = "exp_hyperparameter_on_memory_usage_",
        dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gcn_ggnn_json",
        dir_out = "exp3_thesis_figs/paras"):
    params = yaml.load(open("cfg_file/gcn_ggnn_exp_hds.yaml"))

    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, log_y = params['variables'], params[
        'file_prefix'], params['file_suffix'], params['log_y']
    xlabel = "隐藏向量维度\n" + r"($dim(\mathbf{h}^1_x)$)"

    base_size = 14
    for alg in algs:
        df = pd.read_csv("paper_exp1_super_parameters/memory/" + file_out + alg + '.csv', index_col=0)
        fig, ax = plt.subplots(figsize=(7/2, 7/2), tight_layout=True)
        ax.set_yscale("symlog", basey=2)
        if infer_flag:
            ax.set_ylabel("推理阶段内存使用 (MB)", fontsize=base_size+2)
        else:
            ax.set_ylabel("训练阶段内存使用 (MB)", fontsize=base_size+2)
        ax.set_xlabel(xlabel, fontsize=base_size+2)
        ax.set_xticks(list(range(len(variables))))
        ax.set_xticklabels([str(i) for i in variables], fontsize=base_size, rotation=30)
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            ax.plot(df.index, df[c], marker=markers[i], markersize=7, label=datasets_maps[c])
        ax.legend()
        fig.savefig(dir_out + "/" + file_out + alg + "." + file_type, dpi=400)
        plt.close()


if __name__ == "__main__":
    pic_calculations_gcn_ggnn(file_type="png")
    run_memory_gcn_ggnn(file_type="png")
    pic_calculations_gcn_ggnn("png", infer_flag=True,
        file_prefix = "exp_hyperparameter_on_inference_vertex_edge_phase_time_",
        dir_cal="/gcn_ggnn_inference_exp/hds_exp/calculations/",
        dir_out="exp3_thesis_figs/paras")
