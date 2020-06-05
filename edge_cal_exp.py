import os
import time
import sys
import sqlite3
import numpy as np
import pandas as pd
from utils import get_real_time, get_int


def get_edge_time(cur, outliers, alg='gcn'):
    """
    获取edge-cal细分下的各算子的用时
    :param cur: sqlite的cursor对象
    :param alg: 算法
    :return: ['collect', 'message', 'aggregate', 'update']对应的用时
    """
    labels = ['collect', 'message', 'aggregate', 'update']
    edge_times = []
    for label in labels:
        step = 6 if alg == 'gaan' else 2
        sql = "select start, end, text from nvtx_events where text == '{}'".format(label)
        res = cur.execute(sql).fetchall()[step:]  # 过滤掉warm-up中forward阶段的结果
        cost_time = 0
        for i in range(50):
            if i in outliers: continue
            # epoch_time = forward time + backward time + eval time
            # 1. 获取forward time和eval time
            for j in range(step * 2):
                time = get_real_time(res[step * 2 * i + j], cur)[0]
                cost_time += time
            # 2. 基于forward的标签对应的seq获取backward time
            for j in range(step):
                # 思路：首先得到label的时间段[st, ed]; 然后寻找该时间段中所有的seq, 然后找对应的backward中的seq
                # 2.1 寻找该时间段中所有的seq
                seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                seq_res = cur.execute(seq_sql.format(res[step * 2 * i + j][0], res[step * 2 * i + j][1])).fetchall()

                if not seq_res: # ggnn, flickr; edge-cal, message=0
                    continue

                # 2.2 获取seq的最值，seq为连续分布
                min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                seq_backward_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                end_time = cur.execute(seq_backward_sql.format(min_seq)).fetchone()

                # 2.3 为了将空格时间考虑进去，这里在左边时间进行延伸
                start_time = cur.execute(seq_backward_sql.format(max_seq + 1)).fetchone()
                if start_time:
                    backward_time = get_real_time((start_time[1], end_time[1], label), cur)[0]
                else:
                    start_time = cur.execute(seq_backward_sql.format(max_seq)).fetchone()
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]
                cost_time += backward_time

        cost_time /= 50 - len(outliers) # 基于epochs的平均
        print(label, cost_time)
        edge_times.append(cost_time)
    return edge_times


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
        res = get_edge_time(cur, outliers, alg)
        df[data] = res
    pd.DataFrame(df).to_csv('edge-cal/' + alg + '.csv')


