import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import algs, datasets, algorithms

plt.style.use("ggplot")

time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Forward\nEnd', 'Backward\nEnd', 'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']
dir_name = "C:\\Users\\hikk\\Desktop\\config_exp\\dir_json\\"

for data in datasets:
    allocated_current = {}
    for alg in algs:
        file_path = dir_name + 'config0_' + alg + '_' + data + '.json'
        if not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            res = json.load(f)
            dataload_end = np.array(res['forward_start'][0])
            warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
            layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
            layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
            layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
            layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
            forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
            backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
            eval_end = np.array(res['eval_end']).mean(axis=0)
            all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                 backward_end, layer0_eval, layer1_eval, eval_end])
            all_data /= 1024 * 1024
            all_data = all_data.T # 得到所有的數據

            allocated_current[algorithms[alg]] = all_data[0]

    allocated_current = pd.DataFrame(allocated_current, index=time_labels)
    ax = plt.gca()
    ax.set_ylabel("GPU Memory Usage (MB)")
    ax.set_title(data)
    colors = 'rgbm'
    markers = 'oD^s'
    lines = ['-', '--', '-.', ':']
    for i, c in enumerate(allocated_current.columns):
        allocated_current[c].plot(ax=ax, color=colors[i], marker=markers[i], linestyle=lines[i], label=c, rot=45)
    ax.legend()
    plt.show()
    ax.get_figure().savefig("../memory/" + data + ".png")



