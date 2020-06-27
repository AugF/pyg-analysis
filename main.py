import argparse
import yaml
import os
from epochs_exp import run_epochs_exp
from calculations_exp import run_calculations_exp
from edge_cal_exp import run_edge_cal_exp
from operators_exp import run_operators_exp
from pics import run_stages, run_operators, run_memory


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_yaml', type=str, default='cfg_file/hds_heads_exp.yaml', help="yaml file path")
args = parser.parse_args()

params = yaml.load(open(args.cfg_yaml))
print(params)
print("begin epochs ...")
run_epochs_exp(params)
print("begin calculations ...")
run_calculations_exp(params)
print("begin edge_cal ...")
run_edge_cal_exp(params)
print("begin operators ...")
run_operators_exp(params)

# run_one_operator(params)
print("pic stages ...")
run_stages(params)
print("pic operators ...")
run_operators(params)
print("pic memory ...")
run_memory(params)