import os
import time
import sqlite3
import numpy as np
import pandas as pd
from utils import get_real_time


# 2. 获取stages time
def get_stage_time(cur, outliers):
    stages_times = []
    labels = ['forward', 'backward', 'eval']
    for label in labels:
        sql = "select start, end, text from nvtx_events where text == '{}'".format(label)
        res = cur.execute(sql).fetchall()  #
        if not label == 'eval':  # 去除第一个元素
            res = res[1:]
        cost_time = 0
        for i in range(50):
            if i in outliers: continue
            cost_time += get_real_time(res[i], cur)[0]
        stages_times.append(cost_time / (50 - len(outliers)))
    return stages_times


dir_name = r"C:\\Users\\hikk\\Desktop\\config_exp\\dir_sqlite"

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
        res = get_stage_time(cur, outliers)
        df[data] = res
    pd.DataFrame(df).to_csv('stages/' + alg + '.csv')





