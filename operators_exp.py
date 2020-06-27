import os
import time
import sys
import sqlite3
import numpy as np
import json
from utils import get_real_time


def get_operators_time(cur, outliers):
    """
    返回json文件
    :param cur:
    :param outliers:
    :return:
    """
    operators = {}
    for i in range(50):
        if i in outliers: continue
        sql = "select start, end, text from nvtx_events where text == 'epochs{}'".format(i)
        res = cur.execute(sql).fetchall()[0]

        seq_sql = "select start, end, text from nvtx_events where text like '%seq%' and start >= {} and end <= {}".format(
            res[0], res[1])
        seq_res = cur.execute(seq_sql).fetchall()

        operators_times = {}  # 基本的算子，和其对应的cpu的时间
        ope_sql = 'select text from nvtx_events where start > {} and end < {}'
        for r in seq_res:
            t = cur.execute(ope_sql.format(r[0], r[1])).fetchall()
            if len(t) == 1 and t[0] == ('__stop_profile',):
                oper = r[2].split(',')[0]
                if oper in operators_times.keys():
                    operators_times[oper].append(r)
                else:
                    operators_times[oper] = [r]

        cuda_times = {}  # 基本算子在cuda上运行的时间
        times = 0
        for k in operators_times.keys():
            cuda_times[k] = 0
            for x in operators_times[k]:
                cuda_times[k] += get_real_time(x, cur)[0]
            times += cuda_times[k]

        if operators == {}:  # 第一轮时，算子结果还未知
            operators = cuda_times
        else:
            for k, v in cuda_times.items():
                operators[k] += v

    for k in operators.keys():
        operators[k] /= 50 - len(outliers)
    return operators


# if len(sys.argv) < 2 or sys.argv[1] not in algs:
#     print("python operators_exp.py [alg]")
#     sys.exit(0)

# alg = sys.argv[1]
def run_operators_exp(params):
    dir_name, dir_out, algs, datasets = params['dir_name'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix = params['variables'], params['file_prefix'], params['file_suffix']

    base_path = os.path.join(dir_out, "operators")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for alg in algs:
        for data in datasets:
            for var in variables:
                out_json_path = base_path + "/" + alg + '_' + data + file_prefix + str(var) + file_suffix + ".json"
                if os.path.exists(out_json_path):
                    continue
                outlier_file = dir_out + '/epochs/' + alg + '_' + data + file_prefix + str(var) + file_suffix + '_outliers.txt'
                file_path = dir_name + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.sqlite'
                if not os.path.exists(file_path) or not os.path.exists(outlier_file):
                    continue
                print(file_path)
                cur = sqlite3.connect(file_path).cursor()
                print(data, alg)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
                res = get_operators_time(cur, outliers)

                with open(out_json_path, "w") as f:
                    json.dump(res, f)


def run_one_operator():
    import yaml
    params = yaml.load(open('cfg_file/hidden_dims_3_exp.yaml'))
    dir_name, dir_out, algs, datasets = params['dir_name'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix = params['variables'], params['file_prefix'], params['file_suffix']
    
    base_path = os.path.join(dir_out, "operators")
    alg, data, var = 'gcn', 'amazon-photo', 16
    outlier_file = dir_out + '/epochs/' + alg + '_' + data + file_prefix + str(var) + file_suffix + '_outliers.txt'
    file_path = dir_name + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.sqlite'
    if not os.path.exists(file_path):
        return
    print(file_path)
    cur = sqlite3.connect(file_path).cursor()
    print(data, alg)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
    res = get_operators_time(cur, outliers)
    print(res)


def run_config_exp():
    dir_out = "config_exp"
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_name = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/config_exp/dir_sqlite"
    base_path = os.path.join(dir_out, "operators")
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
            res = get_operators_time(cur, outliers)
            df[data] = res
        pd.DataFrame(df).to_csv(out_path)


run_config_exp()
