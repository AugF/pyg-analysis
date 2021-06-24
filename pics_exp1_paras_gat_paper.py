import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, autolabel, datasets_maps, get_inference_expansion_memory
plt.style.use("ggplot")
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
xlabel = [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"]
             
def pic_calculations_gat(file_type="png", infer_flag=False,
                         file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
                         dir_in = "paper_exp1_super_parameters",
                         dir_cal = '/hidden_dims_exp/',
                         dir_out = "paper_exp1_super_parameters/paras_time_figs",
                         ):
    base_size = 14
    plt.rcParams["font.size"] = base_size
    labels = ['Vertex Calculation', 'Edge Calculation']
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    dir_subs = ["hds_head_dims_exp", "hds_heads_exp"]
    
    for alg in algs:
        fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharey=True, tight_layout=True)
        for k in range(2): # 表示两种情况
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
            for i in range(2):
                ax = axes[i][k]
                ax.set_title(labels[i], fontsize=base_size + 2)
                ax.set_yscale("symlog", basey=2)
                if k == 0:
                    ax.set_ylabel(f"{'Inference' if infer_flag else 'Training'} Time / Epoch (ms)", fontsize=base_size + 2)
                ax.set_xlabel(xlabel[k], fontsize=base_size + 2)
                ax.set_xticks(list(range(len(xticklabels[k]))))
                ax.set_xticklabels(xticklabels[k], fontsize=base_size)
                markers = 'oD^sdp'
                for j, c in enumerate(df[i].columns):
                    ax.plot(df[i].index, df[i][c], marker=markers[j], label=datasets_maps[c])
                # if i == 1 and k == 0:
                if True:
                    ax.legend(ncol=2)
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/" + file_prefix + alg + "." + file_type)
        plt.close()
        
def run_memory_gat(file_type, infer_flag=False, 
                   file_out="exp_hyperparameter_on_memory_usage_", 
                   dir_out="paper_exp1_super_parameters/paras_fig", 
                   dir_memory="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp1_super_parameters/dir_gat_json"):
    plt.rcParams["font.size"] = 12
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    file_prefix = ['_4_', '_']
    file_suffix = ['', '_32']
    
    for alg in algs:
        fig, axes = plt.subplots(1, 2, figsize=(7, 7/2), sharey=True, tight_layout=True)
        for k in range(2):
            df = {}
            for data in datasets:
                df[data] = []
                for var in xticklabels[k]:
                    file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix[k] + str(var) + file_suffix[k] + '.json'
                    if not os.path.exists(file_path):
                        df[data].append(None)
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
            ax = axes[k]
            ax.set_yscale("symlog", basey=2)
            if k == 0:
                ax.set_ylabel(f"{'Inference' if infer_flag else 'Training'} Memory Usage (MB)")
            ax.set_xlabel(xlabel[k])
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels([str(i) for i in xticklabels[k]])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                ax.plot(df.index, df[c], marker=markers[i], label=datasets_maps[c])
            ax.legend()
        fig.savefig(dir_out + "/" + file_out + alg + "." + file_type)
        plt.close()
        

if __name__ == "__main__":
    pic_calculations_gat(file_type="png")
    pic_calculations_gat(file_type="pdf")
    # run_memory_gat(file_type="png")
    # run_memory_gat(file_type="pdf")
    # pic_calculations_gat(file_type="pdf", infer_flag=False,
    #                      file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
    #                      dir_in = "paper_exp1_super_parameters",
    #                      dir_cal = '/hidden_dims_exp/',
    #                      dir_out = "paper_exp1_super_parameters/paras_fig",
    #                      )
    # pic_calculations_gat(file_type="png", infer_flag=False,
    #                      file_prefix = "exp_hyperparameter_on_vertex_edge_phase_time_",
    #                      dir_in = "paper_exp1_super_parameters",
    #                      dir_cal = '/hidden_dims_exp/',
    #                      dir_out = "paper_exp1_super_parameters/paras_fig",
    #                      )
    
    # pic_calculations_gat(file_type="pdf", infer_flag=True,
    #                      file_prefix = "exp_hyperparameter_on_inference_vertex_edge_phase_time_",
    #                      dir_in = "paper_exp1_super_parameters",
    #                      dir_cal = '/gat_inference_exp/',
    #                      dir_out = "paper_exp1_super_parameters/inference_paras_fig",
    #                      )
    # pic_calculations_gat(file_type="png", infer_flag=True,
    #                      file_prefix = "exp_hyperparameter_on_inference_vertex_edge_phase_time_",
    #                      dir_in = "paper_exp1_super_parameters",
    #                      dir_cal = '/gat_inference_exp/',
    #                      dir_out = "paper_exp1_super_parameters/inference_paras_fig",
    #                      )
    # run_memory_gat("pdf", infer_flag=True, 
    #                file_out="exp_hyperparameter_on_inference_memory_usage_", 
    #                dir_out="paper_exp1_super_parameters/inference_paras_fig", 
    #                dir_memory="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp1_super_parameters/dir_gat_inference_json")   
    # run_memory_gat("png", infer_flag=True, 
    #                file_out="exp_hyperparameter_on_inference_memory_usage_", 
    #                dir_out="paper_exp1_super_parameters/inference_paras_fig", 
    #                dir_memory="/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp1_super_parameters/dir_gat_inference_json")   
    
