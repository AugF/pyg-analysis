import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps, dicts
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def pic_others_propogation(label, file_name, file_type):
    algs = ['gaan']
    columns = dicts[label]
    for alg in algs:
        file_path = "paper_exp2_time_break/config_exp/" + label + "/" + alg + ".csv"
        print(file_path)
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns, color_dark2=True)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("Proportion (%)")
        ax.set_ylabel("Dataset")
        plt.tight_layout()
        fig.savefig("paper_exp2_time_break/"+ file_name + alg + "." + file_type) 
        
pic_others_propogation('edge_cal', 'exp_edge_calc_decomposition_', "png")
pic_others_propogation('edge_cal', 'exp_edge_calc_decomposition_', "pdf")