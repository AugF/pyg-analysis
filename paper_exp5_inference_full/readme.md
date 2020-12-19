## Usage

1. config_exp的时间分析结果
```
# epochs
python ../epochs_exp.py cfg_file/config_exp.yaml
# layer_calculations
# png:  pics_exp2_vertex_edge_proportition

# edge_cal pics_edge_cal

# operators

```

2. degrees_exp的时间分析结果
```
# epochs
python ../epochs_exp.py cfg_file/degrees_exp.yaml
# calculations
# png: pics_exp2_vertex_edge_degrees.py
python ../calculations_exp.py cfg_file/degrees_exp.yaml 
```

3. config_memory实验
```
# exp_memory_usage_stage  同pics_exp3_phase.py文件

# exp_memory_expansion_ratio 同pics_exp3_memory_ratio.py文件
```

4. degrees, feats_dims, fix_edges实验
```
# 同pics_exp3对应实验
```