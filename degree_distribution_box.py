import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import re


def draw_graph_box(path, co_to_char, name, savefig_path, tp):
    """画无向图度分布的箱线图"""
    gs = os.listdir(path)
    gs.sort()
    # print(gs)
    data = dict()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        if tp == 1:
            node_degree = [d for n, d in nx.degree(graph)]
        elif tp == 2:
            node_degree = [d for n, d in graph.in_degree()]
        else:
            node_degree = [d for n, d in graph.out_degree()]
        # print(nx.degree(graph))
        # print(node_degree)
        count = re.sub('\\.gml', '', g)
        data[co_to_char[int(count)]] = pd.Series(node_degree)
    # print(data)
    df = pd.DataFrame(data)
    # print(df)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    df.plot.box(showfliers=False, figsize=(62, 22), fontsize=15)
    plt.title(name, fontsize=100)
    plt.xticks(rotation=90)
    plt.grid(linestyle="--", alpha=0.3)
    # print(df.describe())

    plt.savefig(savefig_path + f"{name}.png", format='png', dpi=500)
    # plt.show()


if __name__ == "__main__":
    frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
    rq = frame['开始时间'].drop_duplicates()
    rq = list(rq)
    rq.sort()
    number_to_rq = dict(zip(list(range(len(rq))), rq))
    # gml_paths1 = ['G:/jj_st/bipartite/anjdfen_graphs_gml/graph/', 'G:/jj_st/one_mode_graph/jj/jj_gml/graph/0.1/',
    #               'G:/jj_st/one_mode_graph/st/st_gml/graph/0.1/']
    save_paths = ['G:/jj_st/bipartite/', 'G:/jj_st/one_mode_graph/jj/', 'G:/jj_st/one_mode_graph/st/']
    titles = ['基金-股票网络', '基金网络', '股票网络']
    # for i in range(3):
    #     draw_graph_box(gml_paths1[i], number_to_rq, titles[i] + '度分布', save_paths[i], 1)

    gml_paths2 = ['G:/jj_st/bipartite/anjdfen_graphs_gml/digraph/', 'G:/jj_st/one_mode_graph/jj/jj_gml/digraph/0.2/',
                  'G:/jj_st/one_mode_graph/st/st_gml/digraph/0.2/']
    for i in range(3):
        draw_graph_box(gml_paths2[i], number_to_rq, titles[i] + '入度分布', save_paths[i], 2)
        draw_graph_box(gml_paths2[i], number_to_rq, titles[i] + '出度分布', save_paths[i], 3)
