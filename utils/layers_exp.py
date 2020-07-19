import os
import time
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, survey, algorithms, datasets_maps, datasets, dicts
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12

def get_layers_time(cur, outliers, alg):
    labels = ['layer0', 'layer1', 'loss', 'other']
    layers_time = []

    for label in labels:
        sql = "select start, end, text from nvtx_events where text == '{}'"
        res = cur.execute(sql.format(label)).fetchall()
        cost_time = 0
        if label == 'loss':  # loss_time = forward_time + backward_time
            res = res[1:]
            backward_res = cur.execute(sql.format("backward")).fetchall()[1:]
            for i in range(50):
                if i in outliers: continue
                forward_time = get_real_time(res[i], cur)[0]  # forward time; res[1]

                # cal forward time
                seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                seq_res = cur.execute(seq_sql.format(res[i][0], res[i][1])).fetchall()

                seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                start_time = backward_res[i][0]  # loss结束之处，即为backward开始的时候
                end_time = cur.execute(seq_backward_sql.format(get_int(seq_res[0][0]))).fetchone()[1]
                # 前向传播最小的seq对应于最长的时间

                backward_time = get_real_time((start_time, end_time, label), cur)[0]

                # print(label)
                # print('forward time', forward_time)
                # print('backward time', backward_time)
                cost_time += forward_time + backward_time
        elif label == 'other':  # other
            for i in range(50):
                if i in outliers: continue
                cost_time += get_real_time(res[i], cur)[0]
        else:
            res = res[1:]
            for i in range(50):
                if i in outliers: continue  # 过滤掉异常的情况
                forward_time = get_real_time(res[2 * i], cur)[0]  # forward_time
                eval_time = get_real_time(res[2 * i + 1], cur)[0]  # eval_time

                seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                seq_res = cur.execute(seq_sql.format(res[2 * i][0], res[2 * i][1])).fetchall()

                min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                if start_time:
                    backward_time = get_real_time((start_time[1], end_time[1], label), cur)[0]
                else:
                    start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]

                cost_time += forward_time + backward_time + eval_time

            if alg == 'ggnn':
                if label == 'layer0':  # 对于ggnn, 将input-transform的时间开销加到layer0中
                    input_res = cur.execute(sql.format("input-transform")).fetchall()[1:]
                    for i in range(50):
                        if i in outliers: continue  # 过滤掉异常的情况
                        forward_time = get_real_time(input_res[2 * i], cur)[0]  # forward_time
                        eval_time = get_real_time(input_res[2 * i + 1], cur)[0]  # eval_time

                        seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                        seq_res = cur.execute(seq_sql.format(input_res[2 * i][0], input_res[2 * i][1])).fetchall()

                        min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                        seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                        end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                        start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                        if start_time:
                            backward_time = get_real_time((start_time[1], end_time[1], label), cur)[0]
                        else:
                            start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                            backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]

                        cost_time += forward_time + backward_time + eval_time
                elif label == 'layer1':  # 这里指的是两个layer, 多个layer需要进行修改 todo
                    out_res = cur.execute(sql.format("output-transform")).fetchall()[1:]
                    for i in range(50):
                        if i in outliers: continue  # 过滤掉异常的情况
                        forward_time = get_real_time(out_res[2 * i], cur)[0]  # forward_time
                        eval_time = get_real_time(out_res[2 * i + 1], cur)[0]  # eval_time

                        seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                        seq_res = cur.execute(seq_sql.format(out_res[2 * i][0], out_res[2 * i][1])).fetchall()

                        min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                        seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                        end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                        start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                        if start_time:
                            backward_time = get_real_time((start_time[1], end_time[1], label), cur)[0]
                        else:
                            start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                            backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]

                        cost_time += forward_time + backward_time + eval_time
        cost_time /= 50 - len(outliers)
        print(label, ',', cost_time)
        layers_time.append(cost_time)
    return layers_time


def run_config_exp():
    dir_out = "config_exp"
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_name = "/data/wangzhaokang/wangyunpan/pyg-gnns/config_exp/dir_sqlite"
    base_path = os.path.join(dir_out, "layers")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    for alg in algs:
        df = {}
        out_path = base_path + '/' + alg + '.csv'
        if os.path.exists(out_path):
            continue
        for data in datasets:
            outlier_file = dir_out + '/epochs/' + alg + '_' + data + '_outliers.txt'
            file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
            if not os.path.exists(file_path) or not os.path.exists(outlier_file):
                continue
            cur = sqlite3.connect(file_path).cursor()
            print(data, alg)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
            res = get_layers_time(cur, outliers, alg)
            df[data] = res
        pd.DataFrame(df).to_csv(out_path)

run_config_exp()
