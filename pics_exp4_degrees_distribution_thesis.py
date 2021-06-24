import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.font_manager import _rebuild
_rebuild() 
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)
base_size = 10
plt.rcParams['font.size'] = base_size

graph_path = "amp_graph.npy"
cluster_path = "amp_cluster_90.npy"
graphsage_path = "amp_graphsage_459.npy"

paths = [graph_path, cluster_path, graphsage_path]
names = ['原始图', '聚类采样', '邻居采样']

dir_out = "exp3_thesis_figs/sampling"

def get_degrees_counts(path):
    # 1. 统计出每个节点的度数
    in_degrees = {}
    out_degrees = {}

    graph = np.load(path)
    for i in graph[0, :]:
        if i not in in_degrees.keys():
            in_degrees[i] = 1
        else:
            in_degrees[i] += 1

    for i in graph[1, :]:
        if i not in out_degrees.keys():
            out_degrees[i] = 1
        else:
            out_degrees[i] += 1

    degrees_counts = {}
    for i in in_degrees.values():
        if i not in degrees_counts.keys():
            degrees_counts[i] = 1
        else:
            degrees_counts[i] += 1

    for i in out_degrees.values():
        if i not in degrees_counts.keys():
            degrees_counts[i] = 1
        else:
            degrees_counts[i] += 1
    return degrees_counts


fig, ax = plt.subplots(figsize=(7/2, 5/2), tight_layout=True)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(ymin=0.5, ymax=1e4)
ax.set_xlim(xmin=0.5, xmax=1e4)
ax.set_xlabel("度数", fontsize=base_size+2)
ax.set_ylabel("点数", fontsize=base_size+2)
plt.xticks(fontsize=base_size)
plt.yticks(fontsize=base_size)

markers = 'oD^'
colors = 'rgb'
for i, path in enumerate(paths):
    degrees_counts = get_degrees_counts("paper_exp4_relative_sampling/batch_degrees_distribution/" + path)
    xs = list(degrees_counts.keys())
    ys = [degrees_counts[d] for d in xs]
    ax.scatter(xs, ys, color=colors[i], label=names[i], marker=markers[i])

ax.legend(fontsize=base_size-2)

fig.savefig(dir_out + "/exp_sampling_minibatch_degrees_distribution_amazon-photo.png", dpi=400)