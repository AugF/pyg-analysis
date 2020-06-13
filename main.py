import argparse
import yaml
import os
from epochs_exp import run_epochs_exp
from calculations_exp import run_calculations_exp
from edge_cal_exp import run_edge_cal_exp
from operators_exp import run_operators_exp, run_one_operator
from pics import run_stages, run_operators, run_memory

filename = os.path.join(os.path.dirname(__file__),'test.yaml').replace("\\","/")

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_yaml', type=str, default='cfg_file/hds_heads_exp.yaml', help="yaml file path")
args = parser.parse_args()

params = yaml.load(open(args.cfg_yaml))
print(params)
run_epochs_exp(params)
run_calculations_exp(params)
run_edge_cal_exp(params)
run_operators_exp(params)

# run_one_operator(params)
run_stages(params)
run_operators(params)
run_memory(params)