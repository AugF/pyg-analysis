import time
import os
import sys
import yaml
import math
import sqlite3
import pandas as pd
from utils import get_real_time


# 1. epochs, outliers保存为outlier文件
def get_epoch_time(cur, outlier_file):
    sql = "select start, end, text from nvtx_events where text like 'epochs%'"
    res = cur.execute(sql).fetchall()  # 所有epochs的结果
    if len(res) < 50: # 过滤掉运行异常的sqlite文件
        return None

    epoch_times = [get_real_time(x, cur)[0] for x in res] # 需要单独保存
    tables = {x: i for i, x in enumerate(epoch_times)}

    epoch_times.sort()
    n = len(epoch_times)
    x, y = (n + 1) * 0.25, (n + 1) * 0.75
    tx, ty = math.floor(x), math.floor(y)
    if tx == 0:
        Q1 = epoch_times[tx] * (1 - x + tx)
    elif tx >= n:  # 截断多余部分
        Q1 = epoch_times[tx - 1] * (x - tx)
    else:  # 正常情况
        Q1 = epoch_times[tx - 1] * (x - tx) + epoch_times[tx] * (1 - x + tx)

    if ty == 0:
        Q3 = epoch_times[ty] * (1 - y + ty)
    elif ty >= n:
        Q3 = epoch_times[ty - 1] * (y - ty)
    else:
        Q3 = epoch_times[ty - 1] * (y - ty) + epoch_times[ty] * (1 - y + ty)

    min_val, max_val = Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1)

    outliers = []
    for x in epoch_times:
        if x < min_val or x > max_val:
            outliers.append(tables[x])

    with open(outlier_file, 'w') as f:
        for i in outliers:
            f.write(str(i) + ' ')

    return epoch_times

# if len(sys.argv) < 2 or sys.argv[1] not in datasets:
#     print("python epochs_exp.py flickr")
#     sys.exit(0)

# data = sys.argv[1]

def run_epochs_exp(params):
    dir_name, dir_out, algs, datasets = params['dir_name'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix = params['variables'], params['file_prefix'], params['file_suffix']

    base_path = os.path.join(dir_out, "epochs")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for data in datasets:
        for alg in algs:
            csv_path = base_path + '/' + alg + '_' + data + '.csv'
            if os.path.exists(csv_path): # 断点续传
                continue
            df = {}
            for var in variables:
                outlier_file = base_path + '/' + alg + '_' + data + file_prefix + str(var) + file_suffix + '_outliers.txt'
                file_path = dir_name + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.sqlite'
                if not os.path.exists(file_path):
                    continue
                cur = sqlite3.connect(file_path).cursor()
                print(file_path)
                print(data, alg, var)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                res = get_epoch_time(cur, outlier_file)
                if res == None:
                    print("res is None")
                    continue
                df[var] = res
            pd.DataFrame(df).to_csv(csv_path)


def run_one_file():
    import yaml
    params = yaml.load(open('cfg_file/hds_heads_exp.yaml'))
    dir_name, dir_out, algs, datasets = params['dir_name'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix = params['variables'], params['file_prefix'], params['file_suffix']

    alg, data, var = 'gcn', 'amazon-photo', 16
    csv_path = base_path + '/' + alg + '_' + data + '.csv'
    outlier_file = base_path + '/' + alg + '_' + data + file_prefix + str(var) + file_suffix + '_outliers.txt'
    file_path = dir_name + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.sqlite'
    if os.path.exists(file_path):
        cur = sqlite3.connect(file_path).cursor()
        print(file_path)
        print(data, alg, var)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        res = get_epoch_time(cur, outlier_file)
        print(res)
        

def run_config_exp():
    dir_out = "config_exp"
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/config_exp/dir_sqlite"
    base_path = os.path.join(dir_out, "epochs")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for data in datasets:
        csv_path = base_path + '/' + data + '.csv'
        if os.path.exists(csv_path): # 断点续传
            continue
        df = {}
        for alg in algs:
            outlier_file = base_path + '/' + alg + '_' + data + '_outliers.txt'
            file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
            if not os.path.exists(file_path):
                continue
            cur = sqlite3.connect(file_path).cursor()
            print(file_path)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            res = get_epoch_time(cur, outlier_file)
            if res == None:
                print("res is None")
                continue
            df[alg] = res
        pd.DataFrame(df).to_csv(csv_path)
