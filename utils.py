import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

all_labels = {
        'gcn': ['vertex-cal', 'edge-cal'],
        'gat': ['vertex-cal', 'edge-cal'],
        'ggnn': ['vertex-cal_1', 'vertex-cal_2', 'edge-cal'],
        'gaan': ['vertex-cal', 'edge-cal_attentions', 'edge-cal_gateMax', 'edge-cal_gateMean']
    }


datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
algs = ['gcn', 'ggnn', 'gat', 'gaan']

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

dir_name = r"/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_sqlite"
dir_out = r"hidden_dims_exp"
hds = [16, 32, 64, 128, 256, 512, 1024]

def get_int(str):
    p = 0
    for i, c in enumerate(str[::-1]):
        if not c.isdigit():
            p = i
            break
    return int(str[-p:])


def get_real_time(x, cur): # 对应到cuda的获取真实时间
    # print(x)
    ltime, rtime, text = x
    sql_runtime = "select correlationId from cupti_activity_kind_runtime where start >= {} and end <= {}".format(ltime, rtime)
    event_ids = cur.execute(sql_runtime).fetchall()
    start_time, end_time = sys.maxsize, 0
    for event_id in event_ids:
        for table in ['cupti_activity_kind_kernel', 'cupti_activity_kind_memcpy',
                      'cupti_activity_kind_memset']:
            sql_cuda = "select start, end from {} where correlationId = {} order by start".format(table, event_id[0])
            events = cur.execute(sql_cuda).fetchall()
            if events:  # 有序性
                start_time = min(start_time, events[0][0])
                end_time = max(end_time, events[-1][1])
    if start_time == sys.maxsize or end_time == 0:
        return 0, start_time, end_time
    # print((end_time - start_time) / 1e6, start_time, end_time)
    return (end_time - start_time) / 1e6, start_time, end_time


def survey(labels, data, category_names): # stages, layers, steps，算子可以通用
    for i, c in enumerate(category_names):
        if c[0] == '_':
            category_names[i] = c[1:]

    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, '%.1f' % c, ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax









