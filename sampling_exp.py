import json
import sys
import json
import sqlite3
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

dir_name = "/data/wangzhaokang/wangyunpan/sampling_exp"

algs = ['gcn', 'ggnn', 'gat', 'gaan']
datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

# experiment 1
for data in datasets:
    for alg in algs:
        for mode in modes:
            file_path = osp.join(dir_name, alg + '_' + data + 'graphsage.sqlite')
            if not os.path.exists(file_path):
                print(f"don't have {file_path}")
            cur = sqlite3.connect(file_path).cursor()
            
            