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

# plt.style.use("classic")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (5, 5)), (0, (3, 1, 1, 1))]
base_size = 10
plt.rcParams["font.size"] = base_size

xlabel = ["隐藏向量的维度" + r" $dim(\mathbf{h}^1_x)$" + "\n" + "(#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$" + "\n" + r"(#Head=4, $dim(\mathbf{h}^1_x)$=64)",
          r"#Head" + "\n" + r"($dim(\mathbf{h}^1_x)$=64," + r"$d_a=d_v=d_m$=32)"]


def pic_calculations_gaan(file_type="png", infer_flag=False,
                          file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
                          dir_in = "paper_exp1_super_parameters", 
                          dir_cal = '/gaan_exp/',
                          dir_out = "paper_exp1_super_parameters/paper_figs"
                          ):
    labels = ['点计算', '边计算']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers',
                'coauthor-physics', 'flickr', 'com-amazon']

    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], [
        '8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    dir_paths = ["hds", "hds_d", "hds_head"]

    fig, axs = plt.subplots(2, 3, figsize=(7, 16/3), sharey=True, tight_layout=True)
    for k in range(3):  # 表示两种情况
        dir_path = dir_in + dir_cal + dir_paths[k] + '/calculations'
        df = {}
        df[0] = {}
        df[1] = {}
        for data in datasets:
            file_path = dir_path + '/gaan_' + data + '.csv'  # 这里只与alg和data相关
            if not os.path.exists(file_path):
                continue
            df_t = pd.read_csv(file_path, index_col=0).T
            if df_t.empty:
                # continue
                df[0][data] = [np.nan] * len(xticklabels[k])
                df[1][data] = [np.nan] * len(xticklabels[k])
            else:
                df[0][data] = df_t[0].values.tolist() + [np.nan] * \
                    (len(xticklabels[k]) - len(df_t[0].values))
                df[1][data] = df_t[1].values.tolist() + [np.nan] * \
                    (len(xticklabels[k]) - len(df_t[1].values))

        df[0] = pd.DataFrame(df[0])
        df[1] = pd.DataFrame(df[1])
        for i in [0, 1]:
            ax = axs[i][k]
            ax.set_title(labels[i], fontsize=base_size + 2)
            ax.set_yscale("symlog", basey=2)
            if k == 0:
                if not infer_flag:
                    ax.set_ylabel('平均每轮训练时间 (毫秒)', fontsize=base_size + 2)
                else:
                    ax.set_ylabel('平均每轮推理时间 (毫秒)', fontsize=base_size + 2)              
            ax.set_xlabel(xlabel[k], fontsize=base_size + 2)
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels(xticklabels[k], fontsize=base_size, rotation=30)
            markers = 'oD^sdp'
            for j, c in enumerate(df[i].columns):
                ax.plot(df[i].index, df[i][c], marker=markers[j],
                        markersize=5, label=datasets_maps[c])
            if i == 0 and k == 1:
                ax.legend(ncol=2)
        fig.savefig(dir_out + '/' + file_prefix + "gaan." + file_type)
        plt.close()


def run_memory_gaan(file_type="png", infer_flag=False,
                    file_out = "exp_hyperparameter_on_memory_usage_",
                    dir_out = "paper_exp1_super_parameters/paras_fig", 
                    dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gaan_json"
                    ):
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers',
                'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], [
        '8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    
    dir_paths = ["hds", "hds_d", "hds_head"]

    file_prefix = ['_4_32_', '_4_', '_']
    file_suffix = ['', '_64', '_32_64']

    fig, axes = plt.subplots(
        1, 3, sharey=True, figsize=(7, 8/3), tight_layout=True)
    for k in range(3):
        base_path = os.path.join(dir_out, 'gaan_exp', dir_paths[k], "memory")
        df = {}
        for data in datasets:
            df[data] = []
            for var in xticklabels[k]:
                file_path = dir_memory + '/config0_gaan_' + data + \
                    file_prefix[k] + str(var) + file_suffix[k] + '.json'
                if not os.path.exists(file_path):
                    df[data].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    if infer_flag:
                        df[data].append(get_inference_expansion_memory(res))
                    else:
                        dataload_end = np.array(res['forward_start'][0])
                        warmup_end = np.array(
                            res['forward_start'][1:]).mean(axis=0)
                        layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                        layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                        layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                        layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                        forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                        backward_end = np.array(
                            res['backward_end'][1:]).mean(axis=0)
                        eval_end = np.array(res['eval_end']).mean(axis=0)
                        all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                        all_data /= (1024 * 1024)
                        # 这里记录allocated_bytes.all.max
                        df[data].append(max(all_data[2:, 1]) - all_data[0, 0])

            if df[data] == [None] * (len(xticklabels[k])):
                del df[data]
        df = pd.DataFrame(df)
        df.to_csv("paper_exp1_super_parameters/memory/" + file_out + "gaan_" + dir_paths[k] + ".csv")
        ax = axes[k]
        if log_y:
            ax.set_yscale("symlog", basey=2)
        if k == 0:
            if not infer_flag:
                ax.set_ylabel('训练内存使用 (MB)', fontsize=base_size + 2)
            else:
                ax.set_ylabel('推理内存使用 (MB)', fontsize=base_size + 2) 
        ax.set_xlabel(xlabel[k], fontsize=base_size + 2)
        ax.set_xticks(list(range(len(xticklabels[k]))))
        ax.set_xticklabels([str(i) for i in xticklabels[k]], fontsize=base_size, rotation=30)
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            ax.plot(df.index, df[c], marker=markers[i], markersize=7, label=datasets_maps[c])
        if k == 0:
            ax.legend(fontsize='small', loc='upper left')
        fig.savefig(dir_out + "/" + file_out + "gaan." + file_type)
        plt.close()


if __name__ == "__main__":
    # run_memory_gaan()
    # run_memory_gaan("pdf")
    # pic_calculations_gaan()
    # pic_calculations_gaan("pdf")
    # pic_calculations_gaan(file_type="png", infer_flag=False,
    #                       file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
    #                       dir_in = "paper_exp1_super_parameters", 
    #                       dir_cal = '/gaan_exp/',
    #                       dir_out = "paper_exp1_super_parameters/paras_fig"
    #                       )
    # pic_calculations_gaan(file_type="pdf", infer_flag=False,
    #                       file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
    #                       dir_in = "paper_exp1_super_parameters", 
    #                       dir_cal = '/gaan_exp/',
    #                       dir_out = "paper_exp1_super_parameters/paras_fig"
    #                       )
    # pic_calculations_gaan(file_type="png", infer_flag=True,
    #                       file_prefix = "exp_hyperparameter_on_inference_vertex_edge_phase_time_",
    #                       dir_in = "paper_exp1_super_parameters", 
    #                       dir_cal = '/gaan_inference_exp/',
    #                       dir_out = "paper_exp1_super_parameters/inference_paras_fig"
    #                       )
    # pic_calculations_gaan(file_type="pdf", infer_flag=True,
    #                       file_prefix = "exp_hyperparameter_on_inference_vertex_edge_phase_time_",
    #                       dir_in = "paper_exp1_super_parameters", 
    #                       dir_cal = '/gaan_inference_exp/',
    #                       dir_out = "paper_exp1_super_parameters/inference_paras_fig"
    #                       )
    run_memory_gaan(file_type="png", infer_flag=False,
                    file_out = "exp_hyperparameter_on_memory_usage_",
                    dir_out = "paper_exp1_super_parameters/paper_figs", 
                    dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gaan_json"
                    )
    # run_memory_gaan(file_type="pdf", infer_flag=False,
    #                 file_out = "exp_hyperparameter_on_memory_usage_",
    #                 dir_out = "paper_exp1_super_parameters/paras_fig", 
    #                 dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gaan_json"
    #                 )
    # run_memory_gaan(file_type="png", infer_flag=True,
    #                 file_out = "exp_hyperparameter_on_inference_memory_usage_",
    #                 dir_out = "paper_exp1_super_parameters/inference_paras_fig", 
    #                 dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gaan_inference_json"
    #                 )
    # run_memory_gaan(file_type="pdf", infer_flag=True,
    #                 file_out = "exp_hyperparameter_on_inference_memory_usage_",
    #                 dir_out = "paper_exp1_super_parameters/inference_paras_fig", 
    #                 dir_memory = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp1_super_parameters/dir_gaan_inference_json"
    #                 )

