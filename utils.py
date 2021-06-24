import sys
import numpy as np
import matplotlib.pyplot as plt

all_labels = {
        'gcn': ['vertex-cal', 'edge-cal'],
        'gat': ['vertex-cal', 'edge-cal'],
        'ggnn': ['vertex-cal_1', 'vertex-cal_2', 'edge-cal'],
        'gaan': ['vertex-cal', 'edge-cal_attentions', 'edge-cal_gateMax', 'edge-cal_gateMean']
    }

algorithms = {
    'gcn': 'GCN',
    'ggnn': 'GGNN',
    'gat': 'GAT',
    'gaan': 'GaAN'
}

sampling_modes = {
    'graphsage': 'GraphSAGE',
    'cluster': 'Cluster-GCN'
}

datasets_maps = {
    'amazon-photo': 'amp',
    'pubmed': 'pub',
    'amazon-computers': 'amc',
    'coauthor-physics': 'cph',
    'flickr': 'fli',
    'com-amazon': 'cam'
}

dicts = {
        'stages': ['Forward', 'Backward', 'Eval'],
        'layers': ['Layer0', 'Layer1', 'Loss', 'Other'],
        'calculations': ['Vertex Calculation', 'Edge Calculation'],
        'edge_cal': ['Collection', 'Messaging', 'Aggregation', 'Updating']
    }
# 这里将config实验除外，因为只涉及到两个参数
# hds_heads_exp 参数
datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
dir_out = r"hds_heads_exp"
dir_name = r"/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_head_sqlite"
dir_memory = r"/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_head_json"
variables = [1, 2, 4, 8, 16]
file_prefix = "_" # default _
file_suffix = "_32" # defaulti " "
xlabel = "Heads"



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


def survey(labels, data, category_names, ax=None, color_dark2=False): # stages, layers, steps，算子可以通用
    # labels: 这里是纵轴的坐标; 
    # data: 数据[[], []] len(data[1])=len(category_names); 
    # category_names: 比例的种类
    
    # print("labels", labels, "data", data, "category_names", category_names)
    for i, c in enumerate(category_names):
        if c[0] == '_':
            category_names[i] = c[1:]

    data_cum = data.cumsum(axis=1)
    if color_dark2:
        category_colors = plt.get_cmap('Dark2')(
            np.linspace(0.15, 0.85, data.shape[1]))
    else:
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, data.shape[1]))       
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)
    else:
        fig = None
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.set_xlabel("Proportion (%)")

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
                    color=text_color, fontsize=10)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(-0.1, 1),
              loc='lower left', fontsize=10)

    return fig, ax


# added for pic exp_memory_expansion_ratio;
def autolabel(rects, ax, memory_ratio_flag=False):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if str(height) == 'nan':
            print(str(height))
            ax.text(rect.get_x() + 0.05, 0.41, "Out Of Memory", fontsize=8, rotation=90)
            continue
        #ax.annotate('{}'.format(height),
        #            xy=(rect.get_x() + rect.get_width() / 2, height),
        #           xytext=(0, 3),  # 3 points vertical offset
        #           textcoords="offset points",
        #            ha='center', va='bottom')
        if memory_ratio_flag:
            ax.text(rect.get_x() , height + 1, f"{height:.1f}", fontsize=8)
        else:
            # ax.text(rect.get_x() , height + 0.1, f"{height:.4f}", fontsize=4.5)
            ax.text(rect.get_x() + 0.05 , height + 0.01, f"{height:.2f}", fontsize=8, rotation=90)

    return ax


def get_inference_expansion_memory(res):
    data_memory = res['data load'][0][0]
    max_memory = 0
    print(res.keys())
    for k in res.keys():
        max_memory = max(max_memory, np.array(res[k]).mean(axis=0)[1])
    return (max_memory - data_memory) / (1024 * 1024)




