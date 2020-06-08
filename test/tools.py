import time
import os
import math
import sqlite3
import numpy as np
import pandas as pd
from utils import get_real_time, get_int, all_labels


# 1. epochs, outliers保存为outlier文件
def get_epoch_time(cur, outlier_file):
    sql = "select start, end, text from nvtx_events where text like 'epochs%'"
    res = cur.execute(sql).fetchall()  # 所有epochs的结果
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


# 3. 获取layers time
def get_layers_time(cur, outliers):
    """
    获取epochs中各个的计算
    :param cur: sqlite3的cursor
    :return: labels对应的时间
    """
    labels = ['input-transform', 'layer0', 'layer1', 'output-transform', 'loss', 'other']
    layers_time = []
    for label in labels:
        cost_time = 0
        if label == 'other':
            # other definition:
            #   forward: [log_softmax, foward_end]
            #   backward: [backward_start, log_softmax_Backward]
            #   eval: [log_softmax, eval_end]
            sql = "select start, end, text from nvtx_events where text like '{}'"
            log_res = cur.execute(sql.format('log_softmax%')).fetchall() # warm-up阶段log_softmax算子还不可见
            forward_res = cur.execute(sql.format('forward')).fetchall()[1:]  # remove warm-up epoch
            backward_res = cur.execute(sql.format('backward')).fetchall()[1:]  # remove warm-up epoch
            eval_res = cur.execute(sql.format('eval')).fetchall()
            for i in range(50):
                if i in outliers: continue
                # epoch_time = forward_time + backward_time + eval_time
                forward_time = (get_real_time(forward_res[i], cur)[2] - get_real_time(log_res[2 * i], cur)[1]) / 1e6
                eval_time = (get_real_time(eval_res[i], cur)[2] - get_real_time(log_res[2 * i + 1], cur)[1]) / 1e6

                # 计算others在backward所对应的时间
                id = get_int(log_res[2 * i][2])  # 获取softmax的id
                seq_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                btime = cur.execute(seq_sql.format(id)).fetchone()
                max_time = get_real_time(btime, cur)  # 寻找结束时间
                min_time = get_real_time(backward_res[i], cur)
                backward_time = (max_time[2] - min_time[1]) / 1e6  # 注意检查每一个都要除以1e6

                cost_time += forward_time + backward_time + eval_time
        else:
            sql = "select start, end, text from nvtx_events where text == '{}'".format(label)
            res = cur.execute(sql).fetchall()[1:]  # 过滤
            for i in range(50):
                if i in outliers: continue  # 过滤掉异常的情况
                # 2*i: forward; 2*i+1: eval
                forward_time = get_real_time(res[2 * i], cur)[0]  # forward_time
                eval_time = get_real_time(res[2 * i + 1], cur)[0]  # eval_time

                # 思路：首先得到label的时间段[st, ed]; 然后寻找该时间段中所有的seq, 然后找对应的backward中的seq
                # 2.1 寻找该时间段中所有的seq
                seq_sql = "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'"
                seq_res = cur.execute(seq_sql.format(res[2 * i][0], res[2 * i][1])).fetchall()

                # 2.2 获取seq的最值，seq为连续分布
                min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                # 2.3 寻找对应的backward的seq, 并通过get_real_time()将python用时对应到cuda用时
                seq_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                start_time = cur.execute(seq_sql.format(max_seq)).fetchone()

                # 注：这里寻找的方法 [max_seq[0], (min_seq - 1)[0]], 由于AddBackward0算子的特殊性的原因
                end_time = cur.execute(seq_sql.format(min_seq - 1)).fetchone()
                if end_time:
                    backward_time = get_real_time((start_time[0], end_time[0], label), cur)[0]
                else:
                    end_time = cur.execute(seq_sql.format(min_seq)).fetchone()
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]

                cost_time += forward_time + backward_time + eval_time
        cost_time /= 50 - len(outliers)
        print(label, cost_time)
        layers_time.append(cost_time)
    return layers_time


# 4. vertex-cal, edge-cal: 2*50
def get_cals_time(cur, outliers, labels):
    """
    获取vertex-cal, edge-cal的用时
    :param cur: sqlite的cursor对象
    :param labels: 需要包含vertex的标签在前面， edge的标签在后面
    :return: [vertex-cal time, edge-cal time]
    """
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

                # 2.2 获取seq的最值，seq为连续分布
                min_seq, max_seq = get_int(seq_res[0][0]), get_int(seq_res[-1][0])

                # 2.3 寻找对应的backward的seq, 并通过get_real_time()将python用时对应到cuda用时
                # todo 注意：这里因为torch_scatter算子，发现这里ScatterMax算子是自己写的
                seq_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                start_time = cur.execute(seq_sql.format(max_seq)).fetchone()

                # 注：这里寻找的方法 [max_seq[0], (min_seq - 1)[0]], 由于AddBackward0算子的特殊性的原因
                end_time = cur.execute(seq_sql.format(min_seq - 1)).fetchone()
                if end_time:
                    backward_time = get_real_time((start_time[0], end_time[0], label), cur)[0]
                else:
                    end_time = cur.execute(seq_sql.format(min_seq)).fetchone()
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]
                cost_time += backward_time
        cost_time /= 50 - len(outliers) # 平均epochs
        if 'vertex' in label:
            vertex_time += cost_time
        else:
            edge_time += cost_time
    return [vertex_time, edge_time]


# 5. edge_cal
def get_edge_time(cur, outliers, alg='gcn'):
    """
    获取edge-cal细分下的各算子的用时
    :param cur: sqlite的cursor对象
    :param alg: 算法
    :return: ['collect', 'message', 'aggregate', 'update']对应的用时
    """
    labels = ['collect', 'message', 'aggregate', 'update']
    edge_time = []
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

                # 2.3 寻找对应的backward的seq, 并通过get_real_time()将python用时对应到cuda用时
                seq_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {0}' or text like '%ScatterMax%seq = {0}'"
                start_time = cur.execute(seq_sql.format(max_seq)).fetchone()

                # 注：这里寻找的方法 [max_seq[0], (min_seq - 1)[0]], 由于AddBackward0算子的特殊性的原因
                end_time = cur.execute(seq_sql.format(min_seq - 1)).fetchone()
                if end_time is not None:
                    backward_time = get_real_time((start_time[0], end_time[0], label), cur)[0]
                else:
                    end_time = cur.execute(seq_sql.format(min_seq)).fetchone()
                    print(start_time, end_time)
                    backward_time = get_real_time((start_time[0], end_time[1], label), cur)[0]
                cost_time += backward_time

        cost_time /= 50 - len(outliers) # 基于epochs的平均
        print(label, cost_time)
        edge_time.append(cost_time)
    return edge_time


def get_operators_time(cur, outliers): # todo
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

        print(cuda_times)
        if operators == {}:  # 第一轮时，算子结果还未知
            operators = cuda_times
        else:
            for k, v in cuda_times.items():
                operators[k] += v

    for k in operators.keys():
        operators[k] /= 50 - len(outliers)
    return operators


if __name__ == '__main__':
    dir_name = r"C:\\Users\\hikk\\Desktop\\gnn-parallel-project\\step4-experiment\\config_exp_sqlite"
    # epochs experiment
    datasets = ['flickr', 'com-amazon', 'reddit', 'com-lj']
    algorithms = ['gcn', 'ggnn', 'gat', 'gaan']
    labels = ['stages', 'layers', 'calculations', 'edge-cal']
    for label in ['layers']:
        print(label)
        for alg in ['gat']:
            df = {}
            for data in datasets:
                outlier_file = 'outliers/' + alg + '_' + data + '.txt'
                file_path = dir_name + '/config0_' + alg + '_' + data + '.sqlite'
                if not os.path.exists(file_path):
                    continue
                cur = sqlite3.connect(file_path).cursor()
                print(data, alg)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
                if label == 'stages':
                    res = get_stage_time(cur, outliers)
                elif label == 'layers':
                    res = get_layers_time(cur, outliers)
                elif label == 'calculations':
                    res = get_cals_time(cur, outliers, all_labels[alg])
                elif label == 'edge-cal':
                    res = get_edge_time(cur, outliers, alg)
                else:
                    res = get_epoch_time(cur, outlier_file)
                df[data] = res
            pd.DataFrame(df).to_csv(label + '/' + alg + '.csv')

