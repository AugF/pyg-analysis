import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, datasets_maps
# plt.rcParams["font.size"] = 12

def pic_config_exp_layer_proportion(dir_out="paper_exp6_inference_full", dir_work="paper_exp6_inference_full", file_out="exp_layer_time_proportion_", file_type="png"):
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    columns = ['Layer0-Vertex', 'Layer0-Edge', 'Layer1-Vertex', 'Layer1-Edge']
    # plt.rcParams["font.size"] = 12
    for alg in algs:
        file_path = dir_work + "/config_exp/layers_calculations/" + alg + ".csv"
        df = pd.read_csv(file_path, index_col=0)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns, color_dark2=True)
        ax.set_title(algorithms[alg], loc="right", fontsize=14)
        ax.set_xlabel("Proportion (%)", fontsize=14)
        ax.set_ylabel("Dataset", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.tight_layout()
        fig.savefig(dir_out + "/" + file_out + alg + "." + file_type) 

pic_config_exp_layer_proportion(dir_out="exp_supplement", dir_work="paper_exp5_inference_full",
                                file_out="exp_inference_full_layer_time_proportion_", file_type="png")
pic_config_exp_layer_proportion(dir_out="exp_supplement", dir_work="paper_exp5_inference_full",
                                file_out="exp_inference_full_layer_time_proportion_", file_type="pdf")
pic_config_exp_layer_proportion(dir_out="exp_supplement", dir_work="paper_exp2_time_break",
                                file_out="exp_layer_time_proportion_", file_type="png")
pic_config_exp_layer_proportion(dir_out="exp_supplement", dir_work="paper_exp2_time_break",
                                file_out="exp_layer_time_proportion_", file_type="pdf")