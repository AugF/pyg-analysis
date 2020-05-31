# 单个算法的config文件
import sys
import os
import argparse
import pandas as pd
from utils import getStages, labels

# alg, id作为参数，输入即表示需要取得哪些数
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default='gcn', help='algorithm: [gcn, ggnn, gat, gaan]')
parser.add_argument('--id', type=str, default='epochs', help='label: [epochs, layers, steps, edge_cal]')

args = parser.parse_args()
alg, id = args.alg, args.id
#
dir_config = r'C:\Users\hikk\Desktop\gnn-parallel-project\step4-experiment\config_exp_sqlite'

df = {}
# dataset为纵坐标的名称

for dataset in ['flickr', 'com-amazon', ' reddit', 'com-lj']:
    file_name = 'config0_' + alg + '_' + dataset
    file_path = os.path.join(dir_config, file_name + + '.sqlite')
    if not os.path.exists(file_path):
        continue
    print(file_name) # 真实拥有的数据集
    df[dataset] = getStages(file_path, labels[id])

pd.DataFrame(df).to_csv()
