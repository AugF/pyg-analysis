import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey
plt.style.use("ggplot")


def pic_epochs(datasets, algorithms):
    # 0. pic epochs box

    for i, data in enumerate(datasets):
        ax = plt.gca()
        ax.set_ylabel("ms")
        ax.set_xlabel("algorithm")
        df = pd.read_csv("epochs/" + data + '.csv', index_col=0)
        columns = [algorithms[i] for i in df.columns]
        df.columns = columns
        df.plot(kind='box', title=data, ax=ax)
        plt.show()
        fig = ax.get_figure()
        fig.savefig("epochs/" + data + ".png")
        del ax


def pic_averge_time():
    from utils import datasets
    dir_name = "epochs/"
    all_data = {}
    for data in datasets:
        file_path = dir_name + data + '.csv'
        out = pd.read_csv(file_path, index_col=0)
        all_data[data] = out.mean(axis=0).values
        print(data, all_data[data])

    all_data = {d: all_data[d].tolist() for d in all_data.keys()}
    all_data['coauthor-physics'].append(0.0)
    df = pd.DataFrame(all_data)
    algs = ['GCN', 'GGNN', 'GAT', 'GaAN']
    df.index = algs
    df_t = pd.DataFrame(df.values.T)
    df_t.columns = df.index
    df_t.index = df.columns
    ax = plt.gca()
    ax.set_xlabel('datasets')
    ax.set_ylabel('ms')
    ax.set_title('average epochs time contrast')
    df_t.plot(kind='bar', rot=45, ax=ax)
    plt.show()
    df_t.to_csv("epochs/average_data.csv")
    ax.get_figure().savefig("epochs/average_data.png")


# 1. stages, layers, operators, edge-cal
def pic_stages(label, columns, algorithms):
    for file in os.listdir(label):
        if not file.endswith('.csv'): continue
        file_name = file[:-4]
        df = pd.read_csv(label + "/" + file, index_col=0)
        labels = df.columns
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey(labels, data.T, columns)
        ax.set_title(algorithms[file_name], loc="right")
        ax.set_xlabel("%")
        fig.savefig(label + "/" + file_name + ".png")
        plt.show()
        del ax


def pic_operators():
    for alg in ['gcn', 'ggnn', 'gat', 'gaan']:
        dir_path = '/operators/' + alg + '/'
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
        df.to_csv("operators/" + alg + ".csv")
        columns.append('others')

        fig, ax = survey(df.columns, df.values.T, columns)
        ax.set_title(algorithms[alg], loc="right")
        ax.set_xlabel("%")
        fig.savefig("operators/" + alg + ".png")
        plt.show()
        del ax


if __name__ == '__main__':
    dicts = {
        'stages': ['Forward', 'Backward', 'Eval'],
        'layers': ['Layer0', 'Layer1', 'Loss', 'Other'],
        'calculations': ['Vertex-cal', 'Edge-cal'],
        'edge-cal': ['Collect', 'Message', 'Aggregate', 'Update']
    }
    algorithms = {
        'gcn': 'GCN',
        'ggnn': 'GGNN',
        'gat': 'GAT',
        'gaan': 'GaAN'
    }

    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    # for label in dicts.keys():
    #     pic_stages(label, dicts[label], algorithms)
    pic_stages('layers', dicts['layers'], algorithms)
