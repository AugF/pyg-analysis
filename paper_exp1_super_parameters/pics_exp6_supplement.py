import sqlite3
import os
import numpy as np
import pandas as pd
from epochs_exp import get_epoch_time
from calculations_exp import get_calculations_time

file_names = ["config0_gat_amazon-computers_4_256", "config0_gat_coauthor-physics_4_256",
        "config0_gat_flickr_4_128", "config0_gat_flickr_16_32"]

dir_in = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/exp_supplement"
dir_out = "exp_supplement"

df = {}
# run epochs_exp
for file_name in file_names[:1]:
    outlier_file = dir_out + "/" + file_name + "_outliers.txt"
    print(outlier_file)
    # epochs_time
    if not os.path.exists(outlier_file):
        cur = sqlite3.connect(dir_in + "/dir_sqlite_new/" + file_name + ".sqlite").cursor()
        res = get_epoch_time(cur, outlier_file)
        print(res)
        
    outliers = np.genfromtxt(outlier_file, dtype=np.int).reshape(-1)
    # calculations_exp
    cur = sqlite3.connect(dir_in + "/dir_sqlite_new/" + file_name + ".sqlite").cursor()
    res = get_calculations_time(cur, outliers, "gat", layer=2, infer_flag=False)
    print(file_name, res)
    df[file_name] = res

pd.DataFrame(df, index=["ver_cal", "edge_cal"]).to_csv(dir_out + "/" + "gat_else.csv")

