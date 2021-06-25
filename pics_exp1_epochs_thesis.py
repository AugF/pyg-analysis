# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import algorithms, datasets
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
rcParams.update(config)

dir_in = "paper_exp2_time_break/config_exp/epochs"
algs = ['gcn', 'ggnn', 'gat', 'gaan']
color = dict(medians='red')

def pics_epochs_violin(file_type="png"):
    for data in datasets:
        fig, ax = plt.subplots(figsize=(7/3, 7/3))
        ax.set_ylabel("单轮训练时间 (ms)")
        ax.set_xlabel("算法")
        df = pd.read_csv(dir_in + '/' + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax, color=color)
        plt.tight_layout()
        fig.savefig("exp3_thesis_figs/paras/exp_absolute_training_time_comparison_" + data + "." + file_type, dpi=400)


pics_epochs_violin()