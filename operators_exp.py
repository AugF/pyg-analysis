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


dir_name = r"C:\\Users\\hikk\\Desktop\\config_exp\\dir_sqlite"

algs = ['gaan']
datasets = ['com-amazon']

# if len(sys.argv) < 2 or sys.argv[1] not in algs:
#     print("python operators_exp.py [alg]")
#     sys.exit(0)
#
# alg = sys.argv[1]
for alg in algs:
    for data in datasets:
        outlier_file = 'outliers/' + alg + '_' + data + '.txt'
        file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
        if not os.path.exists(file_path):
            print("sqlite file not exisit!")
            continue

        cur = sqlite3.connect(file_path).cursor()
        print(data, alg)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
        res = get_operators_time(cur, outliers)

        with open("operators/" + alg + '/' + data + ".json", "w") as f:
            json.dump(res, f)


