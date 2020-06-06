from utils import datasets, algorithms, survey, algs
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fun():
    for alg in ['gcn', 'ggnn', 'gat', 'gaan']:
        dir_path = '../operators/' + alg + '/'
        df = {}
        for data in datasets:
            file_path = dir_path + data + ".json"
            if not os.path.exists(file_path):
                continue
            print(file_path)
            with open(file_path) as f:
                operators = json.load(f)
                sum_time = sum(operators.values())
                df[data] = [operators[op] for op in columns[alg]]
                others_time = sum_time - sum(df[data])
                df[data].append(others_time)
        df = pd.DataFrame(df)
        df.to_csv("../operators/" + alg + ".csv")
        labels = df.columns
        data = 100 * df.values / df.values.sum(axis=0)

        fig, ax = survey(labels, data.T, columns[alg] + ['others'])
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("%")
        fig.savefig("../operators/" + alg + ".png")
        plt.show()
        del ax


for alg in algs:
    dir_path = '../operators/' + alg + '/'
    all_percent_ops = {} # 总的percent ops
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
            percent_ops = {k: 100.0 * ops[k] / s for k in ops.keys()} # 先算比例
            all_percent_ops[data] = percent_ops
            if res == {}:
                res = percent_ops.copy()
            else:
                for k in res.keys():
                    res[k] += percent_ops[k]

    res = {k: res[k] / cnt for k in res.keys()} # 对数据集求平均
    res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True) # 排序，选择topk算子
    columns = [i[0] for i in res_sort[:5]]

    df = {} # 获取实际百分比
    for k in all_percent_ops.keys():
        df[k] = []
        for c in columns:
            df[k].append(all_percent_ops[k][c])
        df[k].append(100 - sum(df[k]))

    df = pd.DataFrame(df)
    df.to_csv("../operators/" + alg + ".csv")
    columns.append('others')

    fig, ax = survey(df.columns, df.values.T, columns)
    ax.set_title(algorithms[alg], loc="right")
    ax.set_xlabel("%")
    fig.savefig("../operators/" + alg + ".png")
    plt.show()
    del ax

