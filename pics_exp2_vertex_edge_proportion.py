import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def pic_config_exp_layer_proportion(file_type="png"):
    algs = ['gaan']
    columns = ['Layer0-Vertex', 'Layer0-Edge', 'Layer1-Vertex', 'Layer1-Edge']
    dir_out = "paper_exp2_time_break"
    for alg in algs:
        file_path = "paper_exp2_time_break/config_exp/layers_calculations/" + alg + ".csv"
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns, color_dark2=True)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("Proportion (%)")
        ax.set_ylabel("Dataset")
        plt.tight_layout()
        fig.savefig(dir_out + "/exp_layer_time_proportion_" + alg + "." + file_type) 

pic_config_exp_layer_proportion()
pic_config_exp_layer_proportion("pdf")