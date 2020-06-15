import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables
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


# 1. stages, layers, operators, edge-cal
def pic_stages(label, columns, params):
    dir_out, algs, datasets, xlabel, log_y = params['dir_out'], params['algs'], params['datasets'], params['xlabel'], params['log_y']

    if 'graph' in dir_out: # 为了graph和其他数据集做区分
        rows, cols = 1, 1
    else:
        rows, cols = 2, 3
    dir_path = dir_out + '/' + label #todo 修改标签
    for alg in algs:
        if not 'graph' in dir_out:
            plt.figure(figsize=(12, 8))
        plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig = plt.figure(1)
        i = 1
        for data in datasets:
            file_path = dir_path + '/' + alg + '_' + data + '.csv' # 这里只与alg和data相关
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path, index_col=0).T
            if df.empty:
                continue
            df.columns = columns

            ax = plt.subplot(rows, cols, i)
            ax.set_title(algorithms[alg] + ' ' + data)
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel('Time (ms)')
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(df.index))))
            ax.set_xticklabels(list(df.index))
            markers = 'oD^sdp'
            for j, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[j], label=c, rot=0)
            ax.legend()
            i += 1
        fig.tight_layout() # 防止重叠
        fig.savefig(dir_path + "/" + label + "_" + alg + ".png")
            # plt.show()
        plt.close()


def run_stages(params):
    dicts = {
        'stages': ['Forward', 'Backward', 'Eval'],
        'layers': ['Layer0', 'Layer1', 'Loss', 'Other'],
        'calculations': ['Vertex-cal', 'Edge-cal'],
        'edge-cal': ['Collect', 'Message', 'Aggregate', 'Update']
    }
    for label in ['calculations', 'edge-cal']:
        pic_stages(label, dicts[label], params)


def run_operators(params):
    dir_out, algs, datasets = params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']

    if 'graph' in dir_out: # 为了graph和其他数据集做区分
        rows, cols = 1, 1
    else:
        rows, cols = 2, 3
        
    for alg in algs:
        dir_path = dir_out + '/operators/' + alg + '_'
        all_ops = {}  # 总的percent ops
        res = {}
        cnt = 0
        for data in datasets:
            all_ops[data] = {}
            for var in variables:
                file_path = dir_path + data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    continue
                # print(file_path)
                with open(file_path) as f:
                    ops = json.load(f)
                    if ops == {}:
                        continue
                    s = sum(ops.values())
                    all_ops[data][str(var)] = ops
                    percent_ops = {k: 100.0 * ops[k] / s for k in ops.keys()}  # 先算比例
                    if res == {}:
                        res = percent_ops.copy()
                    else:
                        for k in res.keys():
                            res[k] += percent_ops[k]
                cnt += 1
        if cnt == 0:
            continue
        res = {k: res[k] / cnt for k in res.keys()}  # 对数据集求平均
        res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)  # 排序，选择topk算子
        columns = [i[0] for i in res_sort[:5]]

        if not 'graph' in dir_out:
            plt.figure(figsize=(12, 8))
        plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig = plt.figure(1)
        i = 1
        for data in datasets:
            df = {}
            for k in all_ops[data].keys():
                df[k] = []
                for c in columns: # 排除掉最后一个元素
                    df[k].append(all_ops[data][k][c])
                df[k].append(sum(all_ops[data][k].values()) - sum(df[k]))

            if df == {}:
                continue
            df = pd.DataFrame(df).T
            df.columns = [i[1:] if i[0] == '_' else i for i in columns] + ['other']

            df.to_csv(dir_out + "/operators/" + alg + '_' + data + ".csv")

            ax = plt.subplot(rows, cols, i)
            ax.set_title(algorithms[alg] + ' ' + data)
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel('Time (ms)')
            ax.set_xlabel(xlabel)
            ax.set_xticks(list(range(len(df.index))))
            ax.set_xticklabels(list(df.index))
            markers = 'oD^sdp'
            for j, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[j], label=c, rot=0)
            ax.legend()
            i += 1

        fig.tight_layout() # 防止重叠
        fig.savefig(dir_out + "/operators/operators_" + alg + ".png")
        plt.close()


def run_memory(params):
    dir_memory, dir_out, algs, datasets = params['dir_memory'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Forward\nEnd', 'Backward\nEnd',
                   'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']

    for alg in algs:
        df = {}
        for data in datasets:
            df[data] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df[data].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    # print(file_path)
                    # print(res.keys())
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
                    
                df[data].append(max(all_data[0]))
            if df[data] == [None] * (len(variables)):
                del df[data]

        df = pd.DataFrame(df)
        fig, ax = plt.subplots()
        ax.set_title(algorithms[alg])
        if log_y:
            ax.set_yscale("symlog", basey=2)
        ax.set_ylabel("GPU Memory Usage(MB)")
        ax.set_xlabel(xlabel)
        ax.set_xticks(list(range(len(variables))))
        ax.set_xticklabels([str(i) for i in variables])
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[i], label=c, rot=0)
        ax.legend()
        fig.savefig(base_path + "/memory_" + alg + ".png")
        plt.close()


def run():
    import yaml
    params = yaml.load(open('cfg_file/layer_exp.yaml'))
    run_operators(params)

run()