import json
import sys
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, algorithms, sampling_modes, survey

dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/basic_exp/dir_path"
dir_out = "sampling_exp/basic_exp"
modes = ['graphsage']
algs = ['gcn', 'ggnn', 'gat', 'gaan']
datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

layers_path = os.path.join(dir_out, "layers")
if not os.path.exists(layers_path):
    os.makedirs(layers_path)
    
stages_path = os.path.join(dir_out, "stages")
if not os.path.exists(stages_path):
    os.makedirs(stages_path)

memory_path = os.path.join(dir_out, "memory")
if not os.path.exists(memory_path):
    os.makedirs(memory_path)

def run_layer_time():
    for alg in algs:
        if os.path.exists(layers_path + '/' + alg + '.csv'):
            continue
        df = {}
        for data in datasets:
            file_path = os.path.join(dir_name, alg + '_' + data + '_graphsage.sqlite')
            print(file_path)
            if not os.path.exists(file_path):
                print(file_path + " is not existed!")
                continue
            cur = sqlite3.connect(file_path).cursor()
            
            # get overhead
            overhead_sql = 'select start, end from PROFILER_OVERHEAD'
            overhead_results = cur.execute(overhead_sql).fetchall()
            
            # warmup end is begin time        
            begin_time = cur.execute("select start from nvtx_events where text == 'epochs0'").fetchall()[0][0]

            batch_results = cur.execute("select start, end, text from nvtx_events where text like 'batch%' and start >= '{}'".format(begin_time)).fetchall()

            backward_results = cur.execute("select start, end, text from nvtx_events where text == 'backward' and start >= '{}'".format(begin_time)).fetchall()
            layer0_results = cur.execute("select start, end, text from nvtx_events where text == 'layer0' and start >= '{}'".format(begin_time)).fetchall()
            layer1_results = cur.execute("select start, end, text from nvtx_events where text == 'layer1' and start >= '{}'".format(begin_time)).fetchall()
            
            # print(len(batch_results), len(backward_results), len(layer0_results), len(layer1_results))
            cnt = 0
            layer0_time = layer1_time = 0
            for i, batch_res in enumerate(batch_results):
                st, ed = batch_res[:2]
                flag = False
                for overhead_res in overhead_results: # 去除无效的batch
                    ltime, rtime = overhead_res
                    if (ltime >= st and ltime <= ed) or (rtime >= st and rtime <= ed):
                        flag = True
                        break
                if flag: continue
                
                # forward_time
                layer0_time += get_real_time(layer0_results[i], cur)[0]
                layer1_time += get_real_time(layer1_results[i], cur)[0]
                
                # backward_time
                for j, layer_results in enumerate([layer0_results, layer1_results]):
                    seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                    # print(i, "layer", layer_results[i])
                    seq_res = cur.execute(seq_sql.format(layer_results[i][0], layer_results[i][1])).fetchall()

                    # print(seq_res)
                    # sys.exit(0)
                    min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                    seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                    end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                    start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                    if start_time:
                        backward_time = get_real_time((start_time[1], end_time[1], ""), cur)[0]
                    else:
                        start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                        backward_time = get_real_time((start_time[0], end_time[1], ""), cur)[0]
                    if j == 0:
                        layer0_time += backward_time
                    else:
                        layer1_time += backward_time
            layer0_time /= (len(batch_results) - cnt)
            layer1_time /= (len(batch_results) - cnt)
            df[data] = [layer0_time, layer1_time]
        pd.DataFrame(df).to_csv(layers_path + '/' + alg + '.csv')


# less run_time
def run_stages_time():
    file_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/run_time.log"
    import re
    my_file = open(file_path)
    
    mydicts = {}
    for mode in ['cluster', 'graphsage']:
        mydicts[mode] = {}
        for alg in algs:
            mydicts[mode][alg] = {}
            for data in datasets:
                mydicts[mode][alg][data] = []

    data, mode, alg = '', '', ''
    for line in my_file:
        match_title = re.match(r".*Namespace.*, dataset='(.*)', epochs.*mode='(.*)', model='(.*)'.*", line)
        if match_title:
            data, mode, alg = match_title.group(1), match_title.group(2), match_title.group(3)
        match_content = re.match(r".*sampling time: (.*), other time: (.*), all_tim.*", line)
        if match_content:
            mydicts[mode][alg][data].append([float(match_content.group(1)), float(match_content.group(2))])
    my_file.close()
    for mode in ['cluster', 'graphsage']:
        for alg in algs:
            # print(mode, alg)
            df = {}
            for data in datasets:
                # print(mode, alg, data)
                # print(np.array(mydicts[mode][alg][data]))
                # print(np.array(mydicts[mode][alg][data]).mean(axis=0))
                df[data] = np.array(mydicts[mode][alg][data]).mean(axis=0)
                if data == 'com-amazon' and mode == 'cluster':
                    print(data, mode, df[data])
            pd.DataFrame(df).to_csv(stages_path + '/' + mode + '_' + alg + '.csv')


# memory experiment
def run_memory():
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/sampling_exp/basic_exp/dir_json"
    for mode in ['cluster', 'graphsage']:
        df = {}
        for alg in algs:
            df[alg] = []
            for data in datasets:
                file_path = dir_name + "/" + alg + "_" + data + "_" + mode + ".json"
                if not os.path.exists(file_path):
                    df[alg].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    warmup_end = np.array(res['warmup end']).mean(axis=0)
                    layer0 = np.array(res['layer0'][1:]).mean(axis=0)
                    layer1 = np.array(res['layer1'][1:]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    all_data = np.array([warmup_end, layer0, layer1, forward_end,
                                        backward_end])
                    all_data /= (1024 * 1024)
                    df[alg].append(max(all_data[:, 1]) - all_data[0, 1]) # 这里记录allocated_bytes.all.max
        df = pd.DataFrame(df)
        df.to_csv(memory_path + '/memory_' + mode + '.csv')
        fig, ax = plt.subplots()
        ax.set_title(sampling_modes[mode])
        ax.set_ylabel("GPU Memory Usage(MB)")
        ax.set_xlabel("Datasets")
        ax.set_xticks(list(range(len(datasets))))
        ax.set_xticklabels(datasets)
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[i], label=c, rot=10)
        ax.legend()
        fig.savefig(memory_path + "/memory_" + mode + ".png")
        plt.close()


def pic_stages():
    for alg in algs:
        file_path = 'sampling_exp/basic_exp/layers/' + alg + '.csv'
        df = pd.read_csv(file_path, index_col=0)
        labels = df.columns
        data = 100 * df.values / df.values.sum(axis=0)
        fig, ax = survey(labels, data.T, ['Layer0', 'Layer1'])
        ax.set_title('graphsage_' + algorithms[alg], loc="right")
        ax.set_xlabel("%")
        fig.savefig('sampling_exp/basic_exp/layers/graphsage_' + alg + '.png')
        plt.show()


run_memory()