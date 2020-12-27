import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps, dicts
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def pic_others_propogation(label, file_name, file_type, dir_out="paper_exp2_time_break", dir_work="paper_exp2_time_break"):
    algs = ["gcn", "ggnn", "gat", "gaan"]
    columns = dicts[label]
    for alg in algs:
        file_path = dir_work + "/config_exp/" + label + "/" + alg + ".csv"
        print(file_path)
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns, color_dark2=True)
        ax.set_title(algorithms[alg], loc="right", fontsize=14)
        ax.set_xlabel("Proportion (%)", fontsize=14)
        ax.set_ylabel("Dataset", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        fig.savefig(dir_out + "/"+ file_name + alg + "." + file_type) 
        
pic_others_propogation('edge_cal', 'exp_inference_full_edge_calc_decomposition_', "png",
                       dir_out="exp_supplement", dir_work="paper_exp5_inference_full")
pic_others_propogation('edge_cal', 'exp_inference_full_edge_calc_decomposition_', "pdf",
                       dir_out="exp_supplement", dir_work="paper_exp5_inference_full")
pic_others_propogation('edge_cal', 'exp_edge_calc_decomposition_', "png",
                       dir_out="exp_supplement", dir_work="paper_exp2_time_break")
pic_others_propogation('edge_cal', 'exp_edge_calc_decomposition_', "pdf",
                       dir_out="exp_supplement", dir_work="paper_exp2_time_break")