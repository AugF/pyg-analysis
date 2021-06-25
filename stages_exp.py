import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_real_time, get_int, survey, algorithms, datasets_maps, datasets, dicts
plt.style.use("ggplot")
#plt.rcParams["font.size"] = 12


dir_name = "/mnt/data/wangzhaokang/wangyunpan/pyg-gnns/paper_exp2_time_break/dir_config_sqlite"
dir_out = "/mnt/data/wangzhaokang/wangyunpan/pyg-analysis/paper_exp2_time_break/config_exp"
base_path = os.path.join(dir_out, "stages")
if not os.path.exists(base_path):
    os.makedirs(base_path)

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

def run_config_exp():
    for alg in ['gcn', 'ggnn', 'gat', 'gaan']:
        df = {}
        for data in datasets:
            outlier_file = dir_out + '/epochs/' + alg + '_' + data + '_outliers.txt'
            file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
            if not os.path.exists(file_path) or not os.path.exists(outlier_file):
                continue
            cur = sqlite3.connect(file_path).cursor()
            print(data, alg)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
            res = get_stage_time(cur, outliers)
            df[data] = res
        pd.DataFrame(df).to_csv(base_path + '/' + alg + '.csv')
    
run_config_exp()
