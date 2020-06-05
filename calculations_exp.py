import os
import time
import sys
import sqlite3
import numpy as np
import pandas as pd
from utils import get_real_time, get_int, all_labels


def get_calculations_time(cur, outliers, alg):
    labels = all_labels[alg]
    vertex_time, edge_time = 0, 0
    for label in labels:
        sql = "select start, end, text from nvtx_events where text == '{}'".format(label)
        res = cur.execute(sql).fetchall()[2:]  # 不考虑warm up
        cost_time = 0
        for i in range(50):
            if i in outliers: continue
            # epoch_time = forward time + backward time + eval time
            # 1. 获取forward time和eval time
            for j in range(4):
                time = get_real_time(res[4 * i + j], cur)[0]
                cost_time += time
            # 2. 基于forward的标签对应的seq获取backward time
            for j in range(2):
                # 思路：首先得到label的时间段[st, ed]; 然后寻找该时间段中所有的seq, 然后找对应的backward中的seq
                # 2.1 寻找该时间段中所有的seq
                seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                seq_res = cur.execute(seq_sql.format(res[4 * i + j][0], res[4 * i + j][1])).fetchall()

                min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                if start_time:
                    backward_time = get_real_time((start_time[1], end_time[1], label), cur)[0]
                else:
                    start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]
                cost_time += backward_time

        cost_time /= 50 - len(outliers)  # 平均epochs
        if 'vertex' in label:
            vertex_time += cost_time
        else:
            edge_time += cost_time
    return [vertex_time, edge_time]


dir_name = r"C:\\Users\\hikk\\Desktop\\config_exp\\dir_sqlite"

# if len(sys.argv) < 2 or sys.argv[1] not in ['gcn', 'ggnn', 'gat', 'gaan']:
#     print("lack algorithm")
#     sys.exit(0)
#
# alg = sys.argv[1]

for alg in ['gcn', 'ggnn', 'gat', 'gaan']:
    df = {}
    for data in ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']:
        outlier_file = 'outliers/' + alg + '_' + data + '.txt'
        file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
        if not os.path.exists(file_path):
            continue
        cur = sqlite3.connect(file_path).cursor()
        print(data, alg)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
        res = get_calculations_time(cur, outliers, alg)
        df[data] = res
    pd.DataFrame(df).to_csv('calculations/' + alg + '.csv')


