import json
import os

def get_operators_time():
    dir_name = "operators/gcn"
    gcns = {}

    for file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, file)
        with open(file_path) as f:
            operators = json.load(f)
            sum_times = sum(operators.values())
            operators = sorted(operators.items(), key=lambda x: x[1], reverse=True)
            gcns[file[:-5]] = [sum_times, operators]






