import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

datasets = ["amazon-photo", "pubmed",
            "amazon-computers", "coauthor-physics", "flickr"]
modes = ["cluster", "graphsage"]
algs = ["gcn", "ggnn", "gat", "gaan"]

cluster_batchs = [15, 45, 90, 150, 375, 750]

graphsage_batchs = {
    'amazon-photo': [77, 230, 459, 765, 1913, 3825],
    'pubmed': [198, 592, 1184, 1972, 4930, 9859],
    'amazon-computers': [138, 413, 826, 1376, 3438, 6876],
    'coauthor-physics': [345, 1035, 2070, 3450, 8624, 17247],
    'flickr': [893, 2678, 5355, 8925, 22313, 44625]
}


datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
}

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

xticklabels = ['1%', '3%', '6%', '10%', '25%', '50%', 'FULL']

dir_in = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp6_sampling_acc/log_fix_time_new"
dir_out = "sampling_acc_figs"

def save_acc_to_file():
    for mode in modes:
        for data in datasets:
            df_accs = {}
            for alg in algs:
                df_accs[alg] = []
                if mode == "cluster":
                    for i, cs in enumerate(cluster_batchs):
                        file_path = os.path.join(dir_in, '_'.join(
                            [mode, alg, data, str(cs)]) + ".log")
                        if not os.path.exists(file_path):
                            df_accs[alg].append(np.nan)
                            continue
                        print(file_path)
                        test_acc = 0
                        # 读取日志文件，获取bs_times, bs_accs: dims=100
                        with open(file_path) as f:
                            for line in f:
                                match_line = re.match(
                                    r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                                if match_line:
                                    test_acc = max(test_acc, float(match_line.group(1)))
                        df_accs[alg].append(np.nan if test_acc == 0 else test_acc)
                elif mode == "graphsage":
                    for i, gs in enumerate(graphsage_batchs[data]):
                        file_path = os.path.join(dir_in, '_'.join(
                            [mode, alg, data, str(gs)]) + ".log")
                        if not os.path.exists(file_path):
                            df_accs[alg].append(np.nan)
                            continue
                        print(file_path)
                        test_acc = 0
                        # 读取日志文件，获取bs_times, bs_accs: dims=100
                        with open(file_path) as f:
                            for line in f:
                                match_line = re.match(
                                    r"Batch:.*best_test_acc: (.*), cur_use_time: (.*)s", line)
                                if match_line:
                                    test_acc = max(test_acc, float(match_line.group(1)))
                        df_accs[alg].append(np.nan if test_acc == 0 else test_acc)
            print(df_accs)
            pd.DataFrame(df_accs, index=xticklabels[:-1]).to_csv(dir_out + "/" + mode + "_" + data + ".csv")
            

def pics_relative_batch_acc():
    df_full = pd.read_csv("/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/paper_exp5_paras_acc/acc_res/alg_acc.csv", index_col=0)
    for mode in modes:
        for data in datasets:
            df = pd.read_csv(dir_out + "/" + mode + "_" + data + ".csv", index_col=0)
            
            markers = "oD^sdp"

            fig, ax = plt.subplots(figsize=(7/3, 6/3), tight_layout=True)
            for i, c in enumerate(df.columns):
                ax.plot(xticklabels, list(df[c]) + [df_full.loc[datasets_maps[data], c]], markersize=4, marker=markers[i], label=algorithms[c])

            ax.set_xlabel("Relative Batch Size (%)", fontsize=10)
            ax.set_ylabel("Test Accuracy", fontsize=10)
            ax.set_ylim(0.4, 1)
            ax.set_xticklabels(xticklabels, fontsize=8, rotation=30)
            
            ax.legend(fontsize="x-small", ncol=2)
            fig.savefig(dir_out + f"/exp_{mode}_sampling_accuracy_on_{datasets_maps[data]}.png")
            fig.savefig(dir_out + f"/exp_{mode}_sampling_accuracy_on_{datasets_maps[data]}.pdf")
        

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        if str(height) == 'nan':
            ax.text(rect.get_x() + 0.1, 0.405, "Out of Memory", fontsize=10, rotation=90, horizontalalignment='center')
        else:
            ax.text(rect.get_x() + 0.1, height * 1.005,
                    f"{height:.2f}", fontsize=10, rotation=90,  horizontalalignment='center')
    return ax


if __name__ == "__main__":
    pics_relative_batch_acc()
