import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import algs, dir_out, datasets, algorithms
plt.style.use("ggplot")

def survey(labels, data, category_names, ax): # stages, layers, steps，算子可以通用
    for i, c in enumerate(category_names):
        if c[0] == '_':
            category_names[i] = c[1:]

    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

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
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return ax


def run_epochs():
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig = plt.figure(1)   
    for i, data in enumerate(datasets):

        ax = plt.subplot(2, 3, i + 1)
        ax.set_ylabel("ms")
        ax.set_xlabel("algorithm")
        df = pd.read_csv(dir_out + "/epochs/" + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax)
    
    fig.tight_layout() # 防止重叠    
    fig.savefig(dir_out + "/epochs/all.png")
    plt.close()


# 1. stages, layers, operators, edge-cal
def pic_stages(label, columns):
    dir_path = dir_out + '/' + label
    plt.figure(figsize=(13, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig = plt.figure(1)  
    for i, alg in enumerate(algs):
        file_path = dir_path + "/" + alg + ".csv"
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path, index_col=0)
        labels = df.columns
        data = 100 * df.values / df.values.sum(axis=0)
        
        ax = plt.subplot(2, 2, i + 1)
        ax = survey(labels, data.T, columns, ax)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("%")

    fig.tight_layout() # 防止重叠 
    fig.savefig(dir_path + "/all.png")
    plt.close()

def run_stages():
    dicts = {
        'stages': ['Forward', 'Backward', 'Eval'],
        'layers': ['Layer0', 'Layer1', 'Loss', 'Other'],
        'calculations': ['Vertex-cal', 'Edge-cal'],
        'edge-cal': ['Collect', 'Message', 'Aggregate', 'Update']
    }
    for label in ['edge-cal']:
        pic_stages(label, dicts[label])


def run_operators():
    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig = plt.figure(1)  
    for i, alg in enumerate(algs):
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

        ax = plt.subplot(2, 2, i + 1)
        ax = survey(df.columns, df.values.T, columns, ax)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("%")
    fig.tight_layout() # 防止重叠 
    fig.savefig(dir_out + '/operators/all.png')
    plt.close()


def run_memory():
    dir_name = r"/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/config_exp/dir_json"
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Forward\nEnd', 'Backward\nEnd',
                   'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']

    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig = plt.figure(1)  
    for i, data in enumerate(datasets):
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

        ax = plt.subplot(2, 3, i + 1)
        ax.set_ylabel("GPU Memory Usage (MB)")
        ax.set_title(data)
        colors = 'rgbm'
        markers = 'oD^s'
        lines = ['-', '--', '-.', ':']
        for j, c in enumerate(allocated_current.columns):
            allocated_current[c].plot(ax=ax, marker=markers[j], label=c, rot=45)
        ax.legend()
    fig.tight_layout() # 防止重叠 
    fig.savefig(dir_out + '/memory/memory_all.png')
    plt.close()

run_memory()
