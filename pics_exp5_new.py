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
file_out = "exp_time_comparison_between_training_"
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

for alg in algs:
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(algs))
    width = 0.2
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
            
    i = 0
    for data in datasets:
        ax.bar(x + locations[i] * width, df_train_forward[data], width, color=colors[i], edgecolor='black', hatch="///")
        ax.bar(x + locations[i] * width, df_train_backward[data], width, color=colors[i], edgecolor='black', bottom=df_train_forward[data], hatch='...')
        ax.bar(x + locations[i] * width, df_train_evaluation[data], width, color=colors[i], edgecolor='black', bottom=[df_train_forward[data][i] + df_train_backward[data][i] for i in range(len(algs))], hatch='xxx')
        i += 1
   
# set info
ax.set_ylabel("Absolute Time per Epoch (ms)", fontsize=20)
ax.set_xlabel("Dataset", fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=16)

legend_colors = [Line2D([0], [0], color=c, lw=4) for c in colors]
legend_hatchs = [Patch(facecolor='white', edgecolor='r', hatch='////'), Patch(facecolor='white',edgecolor='r', hatch='....'), Patch(facecolor='white', edgecolor='r', hatch='xxxx')]
ax.legend(legend_colors + legend_hatchs, [algorithms[i] for i in algs] + ['Forward', 'Backward', 'Evaluation'], ncol=2, loc="upper left", fontsize=16)

fig.savefig(dir_out + '/' + file_out + data + ".png")
