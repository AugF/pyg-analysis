import os
import time
import sys
import sqlite3
import numpy as np
from utils import get_real_time

# debug
dir_name = r"C:\\Users\\hikk\\Desktop\\config_exp\\dir_sqlite"

alg, data = 'gcn', 'amazon-photo'
outlier_file = 'outliers/' + alg + '_' + data + '.txt'
file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
if not os.path.exists(file_path):
    print("sqlite file not exisit!")
    sys.exit(0)

cur = sqlite3.connect(file_path).cursor()
print(data, alg)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)

operators = {}
for i in range(50):
    if i in outliers: continue
    print(i)
    sql = "select start, end, text from nvtx_events where text == 'epochs{}'".format(i)
    res = cur.execute(sql).fetchall()[0]

    seq_sql = "select start, end, text from nvtx_events where text like '%seq%' and start >= {} and end <= {}".format(
        res[0], res[1])
    seq_res = cur.execute(seq_sql).fetchall()

    print("seq_res", seq_res, '\n', len(seq_res))
    operators_times = {}  # 基本的算子，和其对应的cpu的时间
    ope_sql = 'select text from nvtx_events where start > {} and end < {}'

    cpu_times = 0
    selected_times = 0
    cnt = 0
    for r in seq_res:
        cpu_times += (r[1] - r[0]) / 1e6
        t = cur.execute(ope_sql.format(r[0], r[1])).fetchall()
        if len(t) == 1 and t[0] == ('__stop_profile',):
            selected_times += (r[1] - r[0]) / 1e6
            cnt += 1
            oper = r[2].split(',')[0]
            if oper in operators_times.keys():
                operators_times[oper].append(r)
            else:
                operators_times[oper] = [r]

    print("cpu times", cpu_times)
    print("selected times", selected_times)
    print(operators_times, '\n', len(operators_times), cnt)
    cuda_times = {}  # 基本算子在cuda上运行的时间
    times = 0
    for k in operators_times.keys():
        cuda_times[k] = 0
        for x in operators_times[k]:
            cuda_times[k] += get_real_time(x, cur)[0]
        print(cuda_times[k])
        times += cuda_times[k]

    if operators == {}:  # 第一轮时，算子结果还未知
        operators = cuda_times
    else:
        for k, v in cuda_times.items():
            operators[k] += v

    print("operators_times", sum(operators.values()))
    print(operators, '\n', len(operators))
    break

# for k in operators.keys():
#     operators[k] /= 50 - len(outliers)


