import pandas as pd
import os

# 输出斜率，判断是否是直线
def get_k(x, y):
    assert len(x) == len(y)
    res = []
    for r in range(len(x)):
        if r > 0:
            res.append((x[r] - x[r - 1]) / (y[r] - y[r-1]))
    return res


# 按csv文件输出
def print_time(file, data_info):
    df = pd.read_csv(file)
    x = df.values
    y = [int(i) for i in df.columns[1:]]
    if x.shape[0] <= 0:
        return
    print(data_info, "vertex_cal", get_k(x[0, 1:], y))
    print(data_info, "edge_cal", get_k(x[1, 1:], y))

# memory文件输出    
def print_memory(x, y, columns, data_info):
    for i in range(x.shape[1]):
        print(data_info + columns[i], get_k(x[:, i], y))


print("\nbegin gaan exp\n")
# 1. 判断gaan的情况
dirs = ['hds', 'hds_d', 'hds_head']
for d in dirs:
    print("\n", d)
    # gaan time
    print("\ntime\n")
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters/gaan_exp/" + d + "/calculations"
    for file in os.listdir(dir_path):
        print_time(os.path.join(dir_path, file), file)

    #gaan memory
    print("\nmemory\n")
    file_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters/gaan_exp/" + d + "/memory/gaan.csv"
    data = pd.read_csv(file_path)
    ys = {
        'hds': [16, 32, 64, 128, 256, 512, 1024, 2048], 
        'hds_d': [8, 16, 32, 64, 128, 256],
        'hds_head': [1, 2, 4, 8, 16]
    }
    print_memory(data.values, ys[d], data.columns, data_info="gaan_")

print("\nbegin gcn, ggnn, gat exp\n")
# 判断gcn, ggnn, gat的结果
dirs = ['hds_exp', 'hds_head_dims_exp', 'hds_heads_exp']
for d in dirs:
    print("\n", d)
    # gaan time
    print("\ntime\n")
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters/hidden_dims_exp/" + d + "/calculations"
    for file in os.listdir(dir_path):
        if file.endswith(".csv") and 'gaan' not in file:
            print_time(os.path.join(dir_path, file), file)

    print("\nmemory\n")
    #gaan memory
    dir_path = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-analysis/paper_exp1_super_parameters/hidden_dims_exp/" + d + "/memory"
    ys = {
        'gcn_hds_exp': [16, 32, 64, 128, 256, 512, 1024, 2048], 
        'ggnn_hds_exp': [16, 32, 64, 128, 256, 512, 1024, 2048], 
        'gat_hds_head_exp': [1, 2, 4, 8, 16],
        'gat_hds_head_dims_exp': [8, 16, 32, 64, 128, 256]
    }
    for alg in ['gcn', 'ggnn', 'gat']:
        if alg + '_' + d not in ys or (alg == 'gat' and d == 'hds_exp'):
            continue
        y = ys[alg + '_' + d]
        file_path = os.path.join(dir_path, alg + ".csv")
        if not os.path.exists(file_path):
            continue
        data = pd.read_csv(file_path)
        print_memory(data.values, y, data.columns, data_info=alg + "_")
    
