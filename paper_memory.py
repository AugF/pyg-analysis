import os
import json
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import survey, algorithms, variables, autolabel, datasets_maps
plt.style.use("ggplot")
plt.rcParams["font.size"] = 12
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def run_memory_file(params):
    dir_memory, dir_out, algs, datasets = params['dir_memory'], params['dir_out'], params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for alg in algs:
        df = {}
        for data in datasets:
            df[data] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df[data].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    # print(file_path)
                    # print(res.keys())
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    all_data /= (1024 * 1024)
                    df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
            if df[data] == [None] * (len(variables)):
                del df[data]
        df = pd.DataFrame(df)
        df.to_csv(base_path + "/" + alg + ".csv")


def run_memory_gcn_ggnn():
    file_out="exp_hyperparameter_on_memory_usage_"
    params = yaml.load(open("cfg_file/gcn_ggnn_exp_hds.yaml"))
    
    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/hidden_dims_exp/dir_json"
    dir_out = "hidden_dims_exp"
    base_path = os.path.join(dir_out, "gcn_ggnn_exp/hds_exp", "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for alg in algs:
        df = {}
        for data in datasets:
            df[data] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df[data].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    # print(file_path)
                    # print(res.keys())
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    all_data /= (1024 * 1024)
                    df[data].append(max(all_data[2:, 1]) - all_data[0, 0]) # 这里记录allocated_bytes.all.max
            
            if df[data] == [None] * (len(variables)):
                del df[data]
        df = pd.DataFrame(df)
        fig, ax = plt.subplots()
        # ax.set_title("GPU Memory Usage")
        if log_y:
            ax.set_yscale("symlog", basey=2)
        ax.set_ylabel("Training Memory Usage (MB)")
        ax.set_xlabel(xlabel)
        ax.set_xticks(list(range(len(variables))))
        ax.set_xticklabels([str(i) for i in variables])
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[i], label=datasets_maps[c], rot=0)
        ax.legend()
        df.to_csv(base_path + '/' + alg + ".csv")
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_out + alg + ".pdf")
        plt.close()


def run_memory_gat():
    file_out="exp_hyperparameter_on_memory_usage_"
    algs = ['gat']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    file_prefix = ['_4_', '_']
    file_suffix = ['', '_32']
    xlabel = {'gat': [r"$d_{head}$ (#Head=4)", r"#Head ($d_{head}$=32)"],
              'gaan': [r"$d_{v}$ and #$d_{a}$ (#Head=4)", r"#Head ($d_{v}$=32, $d_{a}$=32)"] 
             }
    
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/hidden_dims_exp/dir_head_json"
    dir_out = "hidden_dims_exp"
    dir_path = ['hds_head_dims_exp', 'hds_heads_exp']
    for alg in algs:
        plt.figure(figsize=(12, 4.8))
        fig = plt.figure(1)
        for k in range(2):
            base_path = os.path.join(dir_out, 'gat_exp', dir_path[k], "memory")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            df = {}
            for data in datasets:
                df[data] = []
                for var in xticklabels[k]:
                    file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix[k] + str(var) + file_suffix[k] + '.json'
                    if not os.path.exists(file_path):
                        df[data].append(None)
                        continue
                    with open(file_path) as f:
                        res = json.load(f)
                        dataload_end = np.array(res['forward_start'][0])
                        warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                        layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                        layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                        layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                        layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                        forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                        backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                        eval_end = np.array(res['eval_end']).mean(axis=0)
                        all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                            backward_end, layer0_eval, layer1_eval, eval_end])
                        all_data /= (1024 * 1024)
                        df[data].append(max(all_data[2:, 1]) - all_data[0, 0]) # 这里记录allocated_bytes.all.max
            
                if df[data] == [None] * (len(xticklabels[k])):
                    del df[data]
            df = pd.DataFrame(df)
            ax = plt.subplot(1, 2, k + 1)
            # ax.set_title("GPU Memory Usage")
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel("Training Memory Usage (MB)")
            ax.set_xlabel(xlabel[alg][k])
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels([str(i) for i in xticklabels[k]])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=datasets_maps[c], rot=0)
            ax.legend()
            df.to_csv(base_path + '/' + alg + ".csv")
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_out + alg + ".pdf")
        plt.close()

def run_memory_gaan(file_type="png"):
    file_out="exp_hyperparameter_on_memory_usage_"
    algs = ['gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    log_y = True

    xticklabels = [['16', '32', '64', '128', '256', '512', '1024', '2048'], ['8', '16', '32', '64', '128', '256'], ['1', '2', '4', '8', '16']]
    xlabel = [r"Hidden Vector Dimension $dim(\boldsymbol{h}^1)$ (#Head=4, $d_a=d_v=d_m$=32)", r"$d_a, d_v, d_m$(#Head=4, $dim(\boldsymbol{h}^1)$=64)",  r"#Head ($dim(\boldsymbol{h}^1)$=4, $d_a=d_v=d_m$=32)"]
    
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/hidden_dims_exp/dir_gaan_json"
    dir_out = "hidden_dims_exp"
    dir_path = ["hds_exp", "hds_d_exp", "hds_head_exp"]
    
    file_prefix = ['_4_32_', '_4_', '_']
    file_suffix = ['', '_64', '_32_64']
    
    for alg in algs:
        plt.figure(figsize=(18, 4.8))
        fig = plt.figure(1)
        for k in range(3):
            base_path = os.path.join(dir_out, 'gaan_exp', dir_path[k], "memory")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            df = {}
            for data in datasets:
                df[data] = []
                for var in xticklabels[k]:
                    file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix[k] + str(var) + file_suffix[k] + '.json'
                    if not os.path.exists(file_path):
                        df[data].append(None)
                        continue
                    with open(file_path) as f:
                        res = json.load(f)
                        dataload_end = np.array(res['forward_start'][0])
                        warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                        layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                        layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                        layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                        layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                        forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                        backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                        eval_end = np.array(res['eval_end']).mean(axis=0)
                        all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                            backward_end, layer0_eval, layer1_eval, eval_end])
                        all_data /= (1024 * 1024)
                        df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
                if df[data] == [None] * (len(xticklabels[k])):
                    del df[data]
            df = pd.DataFrame(df)
            ax = plt.subplot(1, 3, k + 1)
            # ax.set_title("GPU Memory Usage")
            if log_y:
                ax.set_yscale("symlog", basey=2)
            ax.set_ylabel("Training Memory Usage (MB)")
            ax.set_xlabel(xlabel[k])
            ax.set_xticks(list(range(len(xticklabels[k]))))
            ax.set_xticklabels([str(i) for i in xticklabels[k]])
            markers = 'oD^sdp'
            for i, c in enumerate(df.columns):
                df[c].plot(ax=ax, marker=markers[i], label=datasets_maps[c], rot=0)
            ax.legend()
            df.to_csv(base_path + '/' + alg + ".csv")
        fig.tight_layout() 
        fig.savefig(dir_out + "/" + file_out + alg + "." + file_type)
        plt.close()
        
def run_memory_layer():
    file_out="exp_hyperparameter_on_memory_usage_"
    params = yaml.load(open("cfg_file/layer_exp.yaml"))
    
    algs, datasets = params['algs'], params['datasets']
    variables, file_prefix, file_suffix, xlabel, log_y = params['variables'], params['file_prefix'], params['file_suffix'], params['xlabel'], params['log_y']
    
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/hidden_dims_exp/dir_json"
    dir_out = "layer_exp"
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for alg in algs:
        df = {}
        for data in datasets:
            df[data] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data + file_prefix + str(var) + file_suffix + '.json'
                if not os.path.exists(file_path):
                    df[data].append(None)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    # print(file_path)
                    # print(res.keys())
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    all_data /= (1024 * 1024)
                    df[data].append(max(all_data[:, 1]) - all_data[1, 1]) # 这里记录allocated_bytes.all.max
            
            if df[data] == [None] * (len(variables)):
                del df[data]
        df = pd.DataFrame(df)
        fig, ax = plt.subplots()
        # ax.set_title("GPU Memory Usage")
        if log_y:
            ax.set_yscale("symlog", basey=2)
        ax.set_ylabel("Training Memory Usage (MB)")
        ax.set_xlabel(xlabel)
        ax.set_xticks(list(range(len(variables))))
        ax.set_xticklabels([str(i) for i in variables])
        markers = 'oD^sdp'
        for i, c in enumerate(df.columns):
            df[c].plot(ax=ax, marker=markers[i], label=c, rot=0)
        ax.legend()
        df.to_csv(base_path + '/' + alg + ".csv")
        fig.savefig(dir_out + "/" + file_out + alg + ".pdf")
        plt.close()

# added by wangyunpan, for one pics
def run_memory_ratio_config_single(file_type="png"):
    file_out="exp_memory_expansion_ratio"
    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/config_exp/dir_json"
    dir_out = "config_exp"
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    df = []
    for alg in algs:
        results = []
        for data in datasets:
            file_path = dir_memory + '/config0_' + alg + '_' + data + '.json'
            if not os.path.exists(file_path):
                results.append(np.nan)
                continue
            with open(file_path) as f:
                res = json.load(f)
                print(file_path)
                # print(res.keys())
                dataload_end = np.array(res['forward_start'][0])
                print(dataload_end)
                warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                eval_end = np.array(res['eval_end']).mean(axis=0)
                all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                    backward_end, layer0_eval, layer1_eval, eval_end])
                all_data /= (1024 * 1024)
                print(max(all_data[2:, 1]), "data", all_data[0, 0])
                results.append(max(all_data[2:, 1]) / all_data[0, 0]) # max memory / data loader current 
        df.append(results.copy())
    
    labels = [datasets_maps[d] for d in datasets]
    locations = [-1.5, -0.5, 0.5, 1.5]
    x = np.arange(len(labels))
    width = 0.2
    rects = []
    colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
    fig, ax = plt.subplots()
    i = 0
    for (l, c) in zip(locations, colors):
        rects.append(ax.bar(x + l * width, df[i], width, label=algorithms[algs[i]], color=c))
        i += 1
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for r in rects:
        ax = autolabel(r, ax)
    ax.legend(ncol=2)
    fig.savefig(dir_out + "/memory/" + file_out +  "." + file_type)
    plt.close()


def run_memory_ratio_config():
    file_out="exp_memory_expansion_ratio_dataload_"
    log_y = False    
    xlabel = 'Datasets'
    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    xticklabels = {
        'amazon-photo': 'amp',
        'pubmed': 'pub',
        'amazon-computers': 'amc',
        'coauthor-physics': 'cph',
        'flickr': 'fli',
        'com-amazon': 'cam'}
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/config_exp/dir_json"
    dir_out = "config_exp"
    base_path = os.path.join(dir_out, "memory")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for alg in algs:
        results = []
        labels = []
        for data in datasets:
            file_path = dir_memory + '/config0_' + alg + '_' + data + '.json'
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                res = json.load(f)
                print(file_path)
                # print(res.keys())
                dataload_end = np.array(res['forward_start'][0])
                warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                eval_end = np.array(res['eval_end']).mean(axis=0)
                all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                    backward_end, layer0_eval, layer1_eval, eval_end])
                
                # print(all_data)
                all_data /= (1024 * 1024)
                results.append(max(all_data[:, 1]) / all_data[0, 1]) # 这里记录allocated_bytes.all.peak
                labels.append(xticklabels[data])
        
        fig, ax = plt.subplots()
        ax.set_title(algorithms[alg])
        ax.set_ylabel("Ratio")
        ax.set_xlabel(xlabel)
        plt.bar(labels, results)
        fig.savefig(dir_out + "/memory/" + file_out + alg + ".png")
        plt.close()


def run_memory_factors_dense_feats_dims(file_type="png"):
    file_out="exp_memory_expansion_ratio_input_feature_dimension_"
    log_y = True    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    
    variables = [16, 32, 64, 128, 256, 512]
    xticklabels = variables

    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/dense_feats_exp/dir_json"
    base_path = "memory_factors_exp"
    
    file_prefix, file_suffix = '_', ''
    
    for data in datasets:
        df_ratio = {}
        for alg in algs:
            df_ratio[alg] = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data  + file_prefix + str(var) + file_suffix + '.json'
                print(file_path)
                if not os.path.exists(file_path):
                    df_ratio[alg].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    print(file_path)
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    
                    all_data /= (1024 * 1024)
                    df_ratio[alg].append(max(all_data[2:, 1]) / all_data[0, 0]) # 这里记录allocated_bytes.all.peak
    
        fig, ax = plt.subplots()
        ax.set_ylabel("Expansion Ratio")
        ax.set_xlabel("Input Feature Dimension")
        ax.set_xticklabels([0] + xticklabels)
        df_ratio = pd.DataFrame(df_ratio)
        if log_y:
            ax.set_yscale("symlog", basey=2)
         
        locations = [-1.5, -0.5, 0.5, 1.5]
        x = np.arange(len(variables))
        width = 0.2
        rects = []
        colors = plt.get_cmap('Paired')(np.linspace(0.15, 0.85, len(locations)))
        
        i = 0
        for (col, c) in zip(df_ratio.columns, colors):
            rects.append(ax.bar(x + locations[i] * width, df_ratio[col], width, label=algorithms[algs[i]], color=c))
            i += 1
        ax.legend()
        fig.savefig(base_path + "/" + file_out + data + "." + file_type)

# added in 7.9
def run_memory_factors(file_type="png"):
    #file_out="exp_memory_expansion_ratio_input_graph_number_of_vertices_"
    file_out="exp_memory_expansion_ratio_input_graph_number_of_edges_"
    # file_out="exp_memory_expansion_ratio_input_graph_number_of_vertices_fixed_edge_"
    log_y = False    
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    #datasets = ['amazon-photo', 'pubmed', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    datasets = ['graph']
    
    # xticklabels = [1000, 5000, 10000, 20000, 30000, 40000, 50000]
    # variables = ['1k', '5k', '10k', '20k', '30k', '40k', '50k']
    
    variables = [2, 5, 10, 15, 20, 30, 40, 50, 70]
    xticklabels = variables
    
    dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/graph_degrees_exp/dir_json"
    # dir_memory = "/home/wangzhaokang/wangyunpan/gnns-project/pyg-gnns/graph_nodes_exp/dir_edges_json"
    base_path = "memory_factors_exp"
    xlabel = "Average Degree"
    # xlabel = "Number of Vertices"

    # file_prefix, file_suffix = '_', '_500k'
    #file_prefix, file_suffix = '_', '_20'
    file_prefix, file_suffix = '_10k_', ''
    
    df_peak = {}
    df_ratio = {}
    df_current = {}
    for alg in algs:
        df_peak[alg] = []
        df_ratio[alg] = []
        df_current[alg] = []
        for data in datasets:
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_' + data  + file_prefix + str(var) + file_suffix + '.json'
                print(file_path)
                if not os.path.exists(file_path):
                    df_peak[alg].append(np.nan)
                    df_ratio[alg].append(np.nan)
                    df_current[alg].append(np.nan)
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    print(file_path)
                    # print(res.keys())
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    
                    all_data /= (1024 * 1024 * 1024)
                    df_peak[alg].append(max(all_data[2:, 1]))
                    df_ratio[alg].append(max(all_data[2:, 1]) / all_data[0, 0]) # 这里记录allocated_bytes.all.peak
                    df_current[alg].append(all_data[0, 0])
    
    fig, ax = plt.subplots()
    ax.set_ylabel("Peak Memory Usage (GB)")
    ax.set_xlabel(xlabel)
    
    df_peak = pd.DataFrame(df_peak)
    markers = 'oD^sdp'
    for i, c in enumerate(df_peak.columns):
        ax.plot(xticklabels, df_peak[c], marker=markers[i], label=algorithms[c])
    ax.legend()
    fig.savefig(base_path + "/" + file_out + "peak_memory." + file_type)
    
    fig, ax = plt.subplots()
    ax.set_ylabel("Expansion Ratio")
    ax.set_xlabel(xlabel)
    df_ratio = pd.DataFrame(df_ratio)
    markers = 'oD^sdp'
    for i, c in enumerate(df_ratio.columns):
        ax.plot(xticklabels, df_ratio[c], marker=markers[i], label=algorithms[c])
    ax.legend()
    fig.savefig(base_path + "/" + file_out + "expansion_ratio." + file_type)
    print("df_peak", df_peak)
    #print("df_current", df_current)
    print("df_ratio", df_ratio)

def run_feats_dims():
    xlabel = 'Datasets'
    xticklabels = ['amp', 'pub', 'amc', 'cph', 'fli', 'cam']
    fig, ax = plt.subplots()
    ax.set_title("Features Dims")
    ax.set_ylabel("Ratio")
    ax.set_xlabel(xlabel)
    results = [745, 500, 767, 8415, 500, 32]
    plt.bar(xticklabels, results)
    fig.savefig("config_exp/feats.png")
    plt.close()


def pics_memory_data_graph():
    out_path = "exp_input_{}_on_memory_usage_{}.png"
    variables = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100]
    hds = [16, 32, 64, 128, 256, 512]
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = "degree_exp"
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/graph_degrees_exp/dir_json" 
   
    dir_path = dir_out + '/memory'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for h in hds:
        for alg in algs:
            warmup_results = []
            peak_results = []
            ratio_results = []
            labels = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_graph_50k_' + str(var) + '.json'
                print(file_path)
                if not os.path.exists(file_path):
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    
                    all_data /= (1024 * 1024)
                    peak_results.append(max(all_data[2:, 1])) # 这里记录allocated_bytes.all.peak
                    warmup_results.append(all_data[1, 1])
                    ratio_results.append(max(all_data[2:, 1]) / all_data[1, 1])
                    labels.append(var)
            print(labels)
            print(warmup_results)
            print(peak_results)
            print(ratio_results)

            fig, ax = plt.subplots()
            ax.set_xlabel("Average Degrees of Graph")
            ax.set_ylabel("Training Memory Usage (MB)")
            line1,  = ax.plot(labels, warmup_results, 'ob', label="Warm Up Memory", linestyle='-')
            line2,  = ax.plot(labels, peak_results, "Dg", label="Peak Memory", linestyle='-')
            ax2 = ax.twinx()
            ax2.set_ylabel("GPU Memory Expansion Ratio")
            line3, = ax2.plot(labels, ratio_results, "rs--", label="Ratio")
            plt.legend(handles=[line1, line2, line3], loc="upper left")
            fig.tight_layout()
            fig.savefig(dir_path + "/" + out_path.format(h, alg))
            plt.close()


def pics_memory_data_graph_nodes():# hds
    out_path = "exp_input_{}_on_memory_usage_{}.png"
    variables = [1, 2, 5, 10, 20, 30, 40, 50]
    hds = [16, 32, 64, 128, 256, 512]
    algs = ['gcn', 'ggnn', 'gat', 'gaan']
    dir_out = "degree_exp"
    dir_memory = "/data/wangzhaokang/wangyunpan/pyg-gnns/graph_degrees_exp/dir_json" 
   
    dir_path = dir_out + '/memory'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for h in hds:
        for alg in algs:
            warmup_results = []
            peak_results = []
            ratio_results = []
            labels = []
            for var in variables:
                file_path = dir_memory + '/config0_' + alg + '_graph_50k_' + str(var) + '.json'
                print(file_path)
                if not os.path.exists(file_path):
                    continue
                with open(file_path) as f:
                    res = json.load(f)
                    dataload_end = np.array(res['forward_start'][0])
                    warmup_end = np.array(res['forward_start'][1:]).mean(axis=0)
                    layer0_forward = np.array(res['layer0'][1::2]).mean(axis=0)
                    layer0_eval = np.array(res['layer0'][2::2]).mean(axis=0)
                    layer1_forward = np.array(res['layer1'][1::2]).mean(axis=0)
                    layer1_eval = np.array(res['layer1'][2::2]).mean(axis=0)
                    forward_end = np.array(res['forward_end'][1:]).mean(axis=0)
                    backward_end = np.array(res['backward_end'][1:]).mean(axis=0)
                    eval_end = np.array(res['eval_end']).mean(axis=0)
                    all_data = np.array([dataload_end, warmup_end, layer0_forward, layer1_forward, forward_end,
                                        backward_end, layer0_eval, layer1_eval, eval_end])
                    
                    all_data /= (1024 * 1024)
                    peak_results.append(max(all_data[2:, 1])) # 这里记录allocated_bytes.all.peak
                    warmup_results.append(all_data[1, 1])
                    ratio_results.append(max(all_data[2:, 1]) / all_data[0, 0])
                    labels.append(var)
            print(labels)
            print(warmup_results)
            print(peak_results)
            print(ratio_results)

            fig, ax = plt.subplots()
            ax.set_xlabel("Average Degrees of Graph")
            ax.set_ylabel("Training Memory Usage (MB)")
            line1,  = ax.plot(labels, warmup_results, 'ob', label="Warm Up Memory", linestyle='-')
            line2,  = ax.plot(labels, peak_results, "Dg", label="Peak Memory", linestyle='-')
            ax2 = ax.twinx()
            ax2.set_ylabel("GPU Memory Expansion Ratio")
            line3, = ax2.plot(labels, ratio_results, "rs--", label="Ratio")
            plt.legend(handles=[line1, line2, line3], loc="upper left")
            fig.tight_layout()
            fig.savefig(dir_path + "/" + out_path.format(h, alg))
            plt.close()

# run_memory_ratio_config_single(file_type="pdf")
# run_memory_factors_dense_feats_dims(file_type="pdf")
# run_memory_factors(file_type="pdf")
run_memory_gaan(file_type="pdf")
