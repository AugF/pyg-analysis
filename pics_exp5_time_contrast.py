import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from utils import algorithms, datasets_maps, datasets, autolabel
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

dir_out = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/new_exp_supplement"
dir_training_work = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp2_time_break/config_exp/stages"
dir_inference_work = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp5_inference_full/config_exp/epochs"
datasets = ['amazon-photo', 'pubmed', 'amazon-computers',
                'coauthor-physics', 'flickr', 'com-amazon']
algs = ["gcn", "ggnn", "gat", "gaan"]

df_train_forward, df_train_backward, df_train_evaluation = {}, {}, {} 
df_inference = {}

for data in datasets:
    df_train_forward[data] = []
    df_train_backward[data] = []
    df_train_evaluation[data] = []

# step1 获取df_training的时间
for alg in algs:
    dir_path = dir_training_work + "/" + alg + ".csv"
    df = pd.read_csv(dir_path, index_col=0)
    for data in datasets: # forward, backward, eval
        x = np.array(df[data])
        df_train_forward[data].append(x[0])
        df_train_backward[data].append(x[1])
        df_train_evaluation[data].append(x[2])

# step2 获取df_inference的时间
for data in datasets:
    dir_path = dir_inference_work + "/" + data + ".csv"
    df = pd.read_csv(dir_path, index_col=0)
    df_inference[data] = []
    for alg in algs:
        outlier_file = dir_inference_work + '/' + alg + '_' + data + '_outliers.txt'
        if not os.path.exists(outlier_file):
            df_inference[data].append(np.nan)
            continue
        outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
        all_time, cnt = 0.0, 0
        for i in range(50):
            if i in outliers: continue
            all_time += df[alg][i]
            cnt += 1
        df_inference[data].append(all_time / cnt)
    #print(df_inference[data])

# 处理
df_ratio = {}
for data in datasets:
    df_ratio[data] = []
    for i in range(4):
        x = df_inference[data][i] / (df_train_forward[data][i] + df_train_backward[data][i] + df_train_evaluation[data][i])
        df_ratio[data].append(x)
pd.DataFrame(df_ratio, index=algs).to_csv(dir_out + "/inference_ratio.csv")
        
# 预处理
for d in [df_train_forward, df_train_backward, df_train_evaluation, df_inference]:
    del d['coauthor-physics'][-1]

# 画图, 每个数据集画一张图; 算法作为横轴
file_out = "exp_time_comparison_between_training_inference_"
for data in datasets:
    # print(df_train_forward[data])
    # print(df_inference[data])
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    locations = [-0.5, 0.5]
    if data == "coauthor-physics":
        algs = ["gcn", "ggnn", "gat"]
    else:
        algs = ["gcn", "ggnn", "gat", "gaan"]
    x = np.arange(len(algs))
    width = 0.35
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, 4))
    # colors = plt.get_cmap('RdYlGn')(
    #             np.linspace(0.15, 0.85, 2)) 
    # df_training
    ax.bar(x + locations[0] * width, df_train_forward[data], width, color=colors[0], edgecolor='black', hatch="///")
    ax.bar(x + locations[0] * width, df_train_backward[data], width, color=colors[0], edgecolor='black', bottom=df_train_forward[data], hatch='...')
    ax.bar(x + locations[0] * width, df_train_evaluation[data], width, color=colors[0], edgecolor='black', bottom=[df_train_forward[data][i] + df_train_backward[data][i] for i in range(len(algs))], hatch='xxx')

    # df_inference
    ax.bar(x + locations[-1] * width, df_inference[data], width, color=colors[1], edgecolor='black')
   
    # set info
    ax.set_ylabel("Absolute Time per Epoch (ms)", fontsize=20)
    ax.set_xlabel("Algorithm", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels([algorithms[i] for i in algs], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=16)

    # set legend
    legend_hatchs = [Patch(facecolor=colors[0], edgecolor='black', hatch='///'), 
                     Patch(facecolor=colors[0],edgecolor='black', hatch='...'), 
                     Patch(facecolor=colors[0], edgecolor='black', hatch='xxx'),
                     Line2D([0], [0], color=colors[1], lw=4)]
    ax.legend(legend_hatchs, ['Training Forward', 'Training Backward', 'Training Evaluation', 'Inference'], fontsize=16)
    
    fig.savefig(dir_out + '/' + file_out + data + ".png")
    fig.savefig(dir_out + '/' + file_out + data + ".pdf")
