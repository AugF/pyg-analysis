import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algorithms, datasets_maps, dicts
from matplotlib.font_manager import _rebuild
_rebuild() 
base_size = 12
plt.rcParams["font.size"] = base_size

def survey(labels, data, category_names, ax=None, color_dark2=False): # stages, layers, steps，算子可以通用
    # labels: 这里是纵轴的坐标; 
    # data: 数据[[], []] len(data[1])=len(category_names); 
    # category_names: 比例的种类
    
    # print("labels", labels, "data", data, "category_names", category_names)
    for i, c in enumerate(category_names):
        if c[0] == '_':
            category_names[i] = c[1:]

    data_cum = data.cumsum(axis=1)
    if color_dark2:
        category_colors = plt.get_cmap('Dark2')(
            np.linspace(0.15, 0.85, data.shape[1]))
    else:
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))       
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(7/1.5, 5/1.5), tight_layout=True)
    else:
        fig = None
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.set_xlabel("Proportion (%)")

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, '%.1f' % c, ha='center', va='center',
                    color=text_color, fontsize=base_size-2)
    ax.legend(ncol=2, bbox_to_anchor=(-0.1, 1),
              loc='lower left', fontsize=base_size-2)

    return fig, ax


def pic_others_propogation(label, file_name, file_type, dir_out="paper_exp2_time_break", dir_work="paper_exp2_time_break"):
    algs = ["gcn", "ggnn", "gat", "gaan"]
    columns = dicts[label]
    for alg in algs:
        file_path = dir_work + "/config_exp/" + label + "/" + alg + ".csv"
        df = pd.read_csv(file_path, index_col=0)
        print(df)
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey([datasets_maps[i] for i in df.columns], data.T, columns, color_dark2=True)
        ax.set_title(algorithms[alg], loc="right", fontsize=base_size+2)
        ax.set_xlabel("比例 (%)", fontsize=base_size+2)
        ax.set_ylabel("数据集", fontsize=base_size+2)
        plt.xticks(fontsize=base_size)
        plt.yticks(fontsize=base_size)
        plt.tight_layout()
        fig.savefig(dir_out + "/"+ file_name + alg + "." + file_type) 
        
# pic_others_propogation('edge_cal', 'exp_inference_full_edge_calc_decomposition_', "png",
#                        dir_out="exp_supplement", dir_work="paper_exp5_inference_full")
# pic_others_propogation('edge_cal', 'exp_inference_full_edge_calc_decomposition_', "pdf",
#                        dir_out="exp_supplement", dir_work="paper_exp5_inference_full")
pic_others_propogation('edge_cal', 'exp_edge_calc_decomposition_', "png",
                       dir_out="paper_exp2_time_break/paper_figs", dir_work="paper_exp2_time_break")
# pic_others_propogation('edge_cal', 'exp_edge_calc_decomposition_', "pdf",
#                        dir_out="exp_supplement", dir_work="paper_exp2_time_break")