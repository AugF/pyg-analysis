# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algorithms, datasets
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
dir_in = "paper_exp2_time_break/config_exp/epochs"
dir_out = "paper_exp1_super_parameters"
algs = ['gcn', 'ggnn', 'gat', 'gaan']

def pics_epochs_violin(file_type="png"):
    for data in datasets:
        fig, ax = plt.subplots()
        ax.set_ylabel("Training Time / Epoch (ms)")
        ax.set_xlabel("Algorithm")
        df = pd.read_csv(dir_in + '/' + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax)
        plt.tight_layout()
        fig.savefig(dir_out + "/exp_absolute_training_time_comparison_" + data + "." + file_type)

pics_epochs_violin()
pics_epochs_violin(file_type="pdf")