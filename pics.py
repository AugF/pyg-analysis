import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algs, dir_out, datasets, algorithms
plt.style.use("ggplot")


def run_epochs():
    for data in datasets:
        fig, ax = plt.subplots()
        ax.set_ylabel("ms")
        ax.set_xlabel("algorithm")
        df = pd.read_csv(dir_out + "/epochs/" + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax)
        fig.savefig(dir_out + "/epochs/" + data + ".png")


def pic_averge_time():
    dir_name = dir_out + "/epochs/"
    all_data = {}
    for data in datasets:
        file_path = dir_name + data + '.csv'
        out = pd.read_csv(file_path, index_col=0)
        all_data[data] = out.mean(axis=0).values

    all_data = {d: all_data[d].tolist() for d in all_data.keys()}
    all_data['coauthor-physics'].append(0.0)
    df = pd.DataFrame(all_data)
    algs = ['GCN', 'GGNN', 'GAT', 'GaAN']
    df.index = algs
    df_t = pd.DataFrame(df.values.T)
    df_t.columns = df.index
    df_t.index = df.columns
    fig, ax = plt.subplots()
    ax.set_xlabel('datasets')
    ax.set_ylabel('ms')
    ax.set_title('average epochs time contrast')
    df_t.plot(kind='bar', rot=45, ax=ax)
    # plt.show()
    df_t.to_csv(dir_out + "/epochs/average_data.csv")
    fig.savefig(dir_out + "/epochs/average_data.png")


# 1. stages, layers, operators, edge-cal
def pic_stages(label, columns):
    dir_path = dir_out + '/' + label
    for file in os.listdir(dir_path):
        if not file.endswith('.csv'): continue
        file_name = file[:-4]
        df = pd.read_csv(dir_path + "/" + file, index_col=0)
        labels = df.columns
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey(labels, data.T, columns)
        ax.set_title(algorithms[file_name], loc="right")
        ax.set_xlabel("%")
        fig.savefig(dir_path + "/" + file_name + ".png")
        # plt.show()


def run_stages():
    dicts = {
        'stages': ['Forward', 'Backward', 'Eval'],
        'layers': ['Layer0', 'Layer1', 'Loss', 'Other'],
        'calculations': ['Vertex-cal', 'Edge-cal'],
        'edge-cal': ['Collect', 'Message', 'Aggregate', 'Update']
    }
    for label in ['calculations', 'edge-cal']:
        pic_stages(label, dicts[label])


def run_operators():
    for alg in algs:
        dir_path = dir_out + '/operators/' + alg + '_'
        all_percent_ops = {}  # 总的percent ops
        res = {}
        cnt = 0
        for data in datasets:
            file_path = dir_path + data + '.json'
            if not os.path.exists(file_path):
                continue
            cnt += 1
            with open(file_path) as f:
                ops = json.load(f)
                s = sum(ops.values())
                percent_ops = {k: 100.0 * ops[k] / s for k in ops.keys()}  # 先算比例
                all_percent_ops[data] = percent_ops
                if res == {}:
                    res = percent_ops.copy()
                else:
                    for k in res.keys():
                        res[k] += percent_ops[k]

        res = {k: res[k] / cnt for k in res.keys()}  # 对数据集求平均
        res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)  # 排序，选择topk算子
        columns = [i[0] for i in res_sort[:5]]

        df = {}  # 获取实际百分比
        for k in all_percent_ops.keys():
            df[k] = []
            for c in columns:
                df[k].append(all_percent_ops[k][c])
            df[k].append(100 - sum(df[k]))

        df = pd.DataFrame(df)
        df.to_csv(dir_out + "/operators/" + alg + ".csv")
        columns.append('others')

        fig, ax = survey(df.columns, df.values.T, columns)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("%")
        fig.savefig(dir_out + "/operators/" + alg + ".png")
        # plt.show()


def pic_memory(dir_name):
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Forward\nEnd', 'Backward\nEnd',
                   'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']

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
                all_data = all_data.T  # 得到所有的數據

                allocated_current[algorithms[alg]] = all_data[0]

        allocated_current = pd.DataFrame(allocated_current, index=time_labels)
        fig, ax = plt.subplots()
        ax.set_ylabel("GPU Memory Usage (MB)")
        ax.set_title(data)
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        for i, c in enumerate(allocated_current.columns):
            allocated_current[c].plot(ax=ax, color=colors[i], marker=markers[i], linestyle=lines[i], label=c, rot=45)
        ax.legend()
        # plt.show()
        fig.savefig(base_path + "/" + data + ".png")
