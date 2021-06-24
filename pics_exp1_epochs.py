# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algorithms, datasets
from matplotlib.font_manager import _rebuild
_rebuild() 

# plt.style.use("classic")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams["font.size"] = 12
dir_in = "paper_exp2_time_break/config_exp/epochs"
dir_out = "paper_exp1_super_parameters"
algs = ['gcn', 'ggnn', 'gat', 'gaan']
color = dict(medians='red')

def pics_epochs_violin(file_type="png"):
    for data in datasets:
        fig, ax = plt.subplots(figsize=(7/3, 7/3))
        ax.set_ylabel("每轮训练时间 (毫秒)")
        ax.set_xlabel("算法")
        df = pd.read_csv(dir_in + '/' + data + '.csv', index_col=0)
        print(df)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax, color=color)
        plt.tight_layout()
        fig.savefig(dir_out + "/exp_absolute_training_time_comparison_" + data + "." + file_type)

pics_epochs_violin()
# pics_epochs_violin(file_type="pdf")