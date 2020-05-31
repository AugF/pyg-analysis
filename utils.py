import math
import sys
import sqlite3
import matplotlib.pyplot as plt
plt.style.use('ggplot')

labels = {
    'epochs': ['epochs'],
    'stages': ['forward', 'backward', 'eval'],
    'layers': {
        'gcn': ['input-transform', 'layer0', 'layer1', 'output-transform'],
        'ggnn': ['input-transform', 'layer0', 'layer1', 'output_transform'],
        'gat': ['input-transform', 'layer0', 'layer1', 'output-transform'],
        'gaan': ['input-transform', 'layer0', 'layer1', 'output-transform']
    },
    'steps': {
        'gcn': ['vertex-cal', 'edge-cal'],
        'gat': ['vertex-cal', 'edge-cal'],
        'ggnn': ['vertex-cal_1', 'vertex-cal_2', 'edge-cal'],
        'gaan': ['vertex-cal', 'edge-cal_attentions', 'edge-cal_gateMax', 'edge-cal_gateMean']
    },
    'edge_cal': ['collect', 'message', 'aggregate', 'update']
}


def get_int(str):
    p = 0
    for i, c in enumerate(str[::-1]):
        if not c.isdigit():
            p = i
            break
    return int(str[-p:])


def get_real_time(x, cur): # 对应到cuda的获取真实时间
    ltime, rtime, text = x
    sql_runtime = "select correlationId from cupti_activity_kind_runtime where start >= {} and end <= {}".format(ltime, rtime)
    event_ids = cur.execute(sql_runtime).fetchall()
    start_time, end_time = sys.maxsize, 0
    for event_id in event_ids:
        for table in ['cupti_activity_kind_kernel', 'cupti_activity_kind_memcpy',
                      'cupti_activity_kind_memset']:
            sql_cuda = "select start, end from {} where correlationId = {} order by start".format(table, event_id[0])
            events = cur.execute(sql_cuda).fetchall()
            if events:  # 利用有序性
                start_time = min(start_time, events[0][0])
                end_time = max(end_time, events[-1][1])
    print(start_time, end_time, text)
    return (end_time - start_time) / 1e6, start_time, end_time


def get_average(times): # 对异常数据进行清洗，返回平均值
    if len(times) == 0:
        return 0

    times.sort()
    n = len(times)
    x, y = (n + 1) * 0.25, (n + 1) * 0.75
    tx, ty = math.floor(x), math.floor(y)

    if tx == 0:
        Q1 = times[tx] * (1 - x + tx)
    elif tx >= n: # 截断多余部分
        Q1 = times[tx - 1] * (x - tx)
    else: # 正常情况
        Q1 = times[tx - 1] * (x - tx) + times[tx] * (1 - x + tx)

    if ty == 0:
        Q3 = times[ty] * (1 - y + ty)
    elif ty >= n:
        Q3 = times[ty - 1] * (y - ty)
    else:
        Q3 = times[ty - 1] * (y - ty) + times[ty] * (1 - y + ty)

    min_val, max_val = Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1)
    times = list(filter(lambda x: x >= min_val and x <= max_val, times))
    if len(times) == 0:
        return 0
    return sum(times) / len(times)


def getStages(file_path, labels):  # 获得一个sqlite文件的固定labels的平均用时
    con = sqlite3.connect(file_path)
    cur = con.cursor()

    ans = []
    for stage in labels:
        nvtx_sql = "select start, end from nvtx_events where text == '{}'"
        if stage == 'epochs':
            res = []
            for i in range(50):
                res.extend(cur.execute(nvtx_sql.format(stage + str(i))).fetchall())
        else:
            res = cur.execute(nvtx_sql.format(stage)).fetchall()

        print(stage, len(res))
        if stage in ['epochs', 'eval']:
            ranges = range(50)
        elif stage in ['forward', 'backward']:
            ranges = range(1, 51)
        elif stage in ['input-transform', 'layer0', 'layer1', 'output-transform']:
            ranges = range(1, 101, 2)
        elif 'gaan' in file_path and stage in ['collect', 'message', 'aggregate', 'update']:
            ranges = range(6, 606, 12)
        else:
            ranges = range(2, 202, 4)

        times = []
        for r in ranges:  # 排除第一个
            # 2. 寻找该事件对应的所有event
            sql_runtime = "select correlationId from cupti_activity_kind_runtime where start >= ? and end <= ?"
            event_ids = cur.execute(sql_runtime, res[r]).fetchall()
            start_time, end_time = sys.maxsize, 0
            # 3. 寻找每个event的最值，从而得到start_time和end_time
            for event_id in event_ids:
                for table in ['cupti_activity_kind_kernel', 'cupti_activity_kind_memcpy',
                              'cupti_activity_kind_memset']:
                    sql_cuda = "select start, end from {} where correlationId = {}".format(table,
                                                                                           event_id[0])
                    events = cur.execute(sql_cuda).fetchall()
                    if events: # 利用有序性
                        start_time = min(start_time, events[0][0])
                        end_time = max(end_time, events[-1][1])
                    # event = events.fetchone()
                    # while event:
                    #     start_time = min(start_time, event[0])
                    #     end_time = max(end_time, event[1])
                    #     event = events.fetchone()
            print((end_time - start_time) / 1e6)
            times.append((end_time - start_time) / 1e6)
        print(len(times), times)
        ans.append(get_average(times))
    return ans


if __name__ == '__main__':
    # test1: 所有文件是否正常
    # test2: 所有标签是否正常
    # test3: 函数是否正常
    file_path = r"C:\Users\hikk\Desktop\gnn-parallel-project\step4-experiment\config_exp_sqlite/config0_gcn_flickr.sqlite"
    labels = ['epochs']
    getStages(file_path, labels)








