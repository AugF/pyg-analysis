import os
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
        print(labels)
        print(data.T)
        print(columns)
        fig, ax = survey(labels, data.T, columns)
        ax.set_title(algorithms[file_name], loc="right")
        ax.set_xlabel("%")
        fig.savefig(file_name + ".png")
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
