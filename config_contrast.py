import sys
import sqlite3
from utils import getStages, get_real_time

file_path = r"C:\\Users\\hikk\\Desktop\\gnn-parallel-project\\step4-experiment\\config_exp_sqlite\\config0_gcn_flickr.sqlite"

con = sqlite3.connect(file_path)
cur = con.cursor()

lable = 'layer0'


def get_real_time(ltime, rtime): # 对应到cuda的各个算子
    sql_runtime = "select correlationId from cupti_activity_kind_runtime where start >= {} and end <= {}".format(ltime, rtime)
    event_ids = cur.execute(sql_runtime).fetchall()
    start_time, end_time = sys.maxsize, 0
    for event_id in event_ids:
        for table in ['cupti_activity_kind_kernel', 'cupti_activity_kind_memcpy',
                      'cupti_activity_kind_memset']:
            sql_cuda = "select start, end from {} where correlationId = {}".format(table,
                                                                                   event_id[0])
            events = cur.execute(sql_cuda).fetchall()
            if events:  # 利用有序性
                start_time = min(start_time, events[0][0])
                end_time = max(end_time, events[-1][1])
    return (end_time - start_time) / 1e6


# 50 epochs: 去掉warm up的计数
res = cur.execute("select start, end from nvtx_events where text == {}".format(lable)).fetchall()[1:]

# 统计时，直接将warmup给去掉, 即去掉第一个值
# layer以上级别 [1: ]
# layer级别: [2: ]
# edge_cal级别: [2 * (1, 3): ]

cost_time = []

# layers
for i, x in enumerate(res):
    ltime, rtime = x[:2]
    cost_time.append(get_real_time(ltime, rtime)) # forward_time + eval_time
    if i % 2 != 0: # eval阶段, 不用寻找backward_time
        continue
    backward_time = 0
    seq_texts = cur.execute("select text from nvtx_events where start >= {} and end <= {}".format(ltime, rtime)).fetchall()
    # seq
    min_seq, max_seq = seq_texts[0][0][-2:], seq_texts[-1][0][-2:]
    for seq_id in range(min_seq, max_seq + 1):
        # backward_times
        bltime, brtime = cur.execute("select start, end from nvtx_events where text like '%Backward%seq = {}%'".format(seq_id)).fetchone()
        backward_time += get_real_time(bltime, brtime)
    cost_time.append(backward_time)

