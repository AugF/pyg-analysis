import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algs, dir_out, datasets, algorithms, hds
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
        df = pd.read_csv(dir_path + "/" + file, index_col=0).T
        df = pd.DataFrame(np.log(df.values), index=df.index, columns=columns)

        fig, ax = plt.subplots()
        ax.set_title(file_name)
        ax.set_xlabel("hidden dims")
        df.plot(kind="line", ax=ax, marker='*', rot=0)
        fig.savefig(dir_path + "/" + file_name + ".png")
        # plt.show()
        plt.close()


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
            all_percent_ops[data] = {}
            for hd in hds:
                file_path = dir_path + data + '_' + str(hd) + '.json'
                if not os.path.exists(file_path):
                    continue
                cnt += 1
                with open(file_path) as f:
                    ops = json.load(f)
                    s = sum(ops.values())
                    all_percent_ops[str(hd)] = ops
                    percent_ops = {k: 100.0 * ops[k] / s for k in ops.keys()}  # 先算比例
                    if res == {}:
                        res = percent_ops.copy()
                    else:
                        for k in res.keys():
                            res[k] += percent_ops[k]
        if cnt == 0:
            continue
        res = {k: res[k] / cnt for k in res.keys()}  # 对数据集求平均
        res_sort = sorted(res.items(), key=lambda x: x[1], reverse=True)  # 排序，选择topk算子
        columns = [i[0] for i in res_sort[:5]]
        columns.append('other')
        for data in datasets:
            df = {}
            for k in all_percent_ops[data].keys():
                df[k] = []
                for c in columns[:-1]: # 排除掉最后一个元素
                    df[k].append(all_percent_ops[data][k][c])
                df[k].append(sum(all_percent_ops[data][k].values()) - sum(df[k]))

            df = pd.DataFrame(df).T
            df.columns = columns

            for i, c in enumerate(columns):
                if c[0] == '_':
                    df.columns[i] = columns[i][1:]

            df.to_csv(dir_out + "/operators/" + alg + '_' + data + ".csv")
            columns.append('others')

            fig, ax = plt.subplots()
            ax.set_title(algorithms[alg] + ' ' + data)
            ax.set_ylabel('ms')
            ax.set_xlabel("Hidden Dims")
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=c, rot=45)
            ax.legend()
            df.plot(kind="line", ax=ax, marker='*', rot=0) # todo 修改marker, 设置对数轴
            fig.savefig(dir_out + "/operators/" + alg + '_' + data + ".png")
            plt.close()


def run_memory():
    dir_name = r"/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_json"
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    time_labels = ['Data\nLoad', 'Warm\nUp', 'Forward\nLayer0', 'Forward\nLayer1', 'Forward\nEnd', 'Backward\nEnd',
                   'Eval\nLayer0', 'Eval\nLayer1', 'Eval\nEnd']


    for alg in ['gcn', 'ggnn']:
        df = {}
        for data in datasets:
            df[data] = []
            for hd in hds:
                file_path = dir_name + '/config0_' + alg + '_' + data + '_' + str(hd) + '.json'
                if not os.path.exists(file_path):
                    df[data].append(0)
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
                    
                df[data].append(np.log(max(all_data[0])))

        df = pd.DataFrame(df, index=[str(i) for i in hds])
        fig, ax = plt.subplots()
        ax.set_ylabel("GPU Memory Usage")
        ax.set_title(alg)
        df.plot(kind='line', ax=ax)
        fig.savefig(base_path + "/" + alg + ".png")
        plt.close()
