import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import undirected_network as un
from networkx.algorithms import bipartite
import re


def Diameter(folder_path1, folder_path2, pic_name):
    """有向二分图直径在二十年间的变化"""
    d_list = []
    files = os.listdir(folder_path1)
    files.sort()
    for file in files:
        print(file)
        graph = nx.read_gml(folder_path1 + file)
        print(graph)
        # print(graph.is_directed())
        d = un.diameter(graph)
        d_list.append(d)
    dtf = pd.DataFrame({'时间': rq, '直径': d_list})
    dtf.to_excel(folder_path2 + "直径.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(d_list) + 1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, d_list)
    plt.savefig(folder_path2 + pic_name + '.png', format='png', dpi=500)


def Ave_degree(folder_path1, folder_path2, pic_name):
    """计算80个季度的平均度并绘制折线图"""
    ave_degree_list = []
    gs = os.listdir(folder_path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(folder_path1 + g)
        a_d = un.average_degree(graph)
        # print(a_d)
        ave_degree_list.append(a_d)
    # a_s = pd.Series(ave_degree_list, index=rq)
    dtf = pd.DataFrame({'时间': rq, '平均度': ave_degree_list})
    dtf.to_excel(folder_path2 + "平均度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_degree_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, ave_degree_list)
    # a_s.plot(rot='60')
    plt.savefig(folder_path2 + pic_name + '.png', format='png', dpi=500)


def Densities(folder_path1, folder_path2, pic_name):
    """计算80个季度的密度并绘制折线图"""
    den_list = []
    gs = os.listdir(folder_path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(folder_path1 + g)
        top_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
        den = bipartite.density(graph, top_nodes)
        # print(a_d)
        den_list.append(den)
    dtf = pd.DataFrame({'时间': rq, '密度': den_list})
    dtf.to_excel(folder_path2 + "密度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(den_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, den_list)
    # a_s.plot(rot='60')
    plt.savefig(folder_path2 + pic_name + '.png', format='png', dpi=500)


def Clustering(folder_path1, folder_path2, pic_name):
    """计算80个季度的平均聚类系数并绘制折线图"""
    ave_cluster_list = []
    gs = os.listdir(folder_path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(folder_path1 + g)
        ave_cluster = bipartite.average_clustering(graph)
        # print(a_d)
        ave_cluster_list.append(ave_cluster)
    dtf = pd.DataFrame({'时间': rq, '平均聚类系数': ave_cluster_list})
    dtf.to_excel(f"{folder_path2}/平均聚类系数.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_cluster_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, ave_cluster_list)
    # a_s.plot(rot='60')
    plt.savefig(folder_path2 + pic_name + '.png', format='png', dpi=500)


def Assortativity(folder_path1, folder_path2, pic_name):
    """计算80个季度的同配性并绘制折线图"""
    r_list = []
    gs = os.listdir(folder_path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(folder_path1 + g)
        r = nx.degree_assortativity_coefficient(graph)
        r_list.append(r)
    dtf = pd.DataFrame({'时间': rq, '同配系数': r_list})
    dtf.to_excel(folder_path2 + "同配系数.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(r_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, r_list)
    # a_s.plot(rot='60')
    plt.savefig(path2 + pic_name + '.png', format='png', dpi=500)


path1 = "G:/jj_st/bipartite/anjdfen_graphs_gml/digraph/"
path2 = "G:/jj_st/bipartite/有向图/"
path3 = "G:/jj_st/偏好连接选择/网络/t/"
path4 = "G:/jj_st/偏好连接选择2/网络/"
if __name__ == "__main__":
    # 日期
    # frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
    # rq = frame['开始时间'].drop_duplicates()
    # rq = list(rq)
    # rq.sort()

    # 基金股票网络有向图直径
    # Diameter(path1, path2, "基金-股票网络有向图直径")
    # # 平均度
    # Ave_degree(path1, path2, "基金-股票网络有向图平均度")
    # # 密度
    # Densities(path1, path2, "基金-股票网络有向图密度")
    # # 平均聚类系数
    # Clustering(path1, path2, "基金-股票网络有向图平均聚类系数")
    # # 同配系数
    # Assortativity(path1, path2, "基金-股票网络有向图同配系数")

    true_nets = ['07.gml', '21.gml', '64.gml']
    model_nets = os.listdir(path3)
    model_nets.sort()
    for _ in range(3):
        true_net = nx.read_gml(path1 + true_nets[_])
        model_net = nx.read_gml(path3 + model_nets[_])

        print(f"真实网络规模：{true_net}".ljust(60), end=' ')
        print(f"构造网络规模：{model_net}".ljust(60))
        print(f"真实网络直径：{un.diameter(true_net)}".ljust(60), end=' ')
        print(f"构造网络直径：{un.diameter(model_net)}".ljust(60))
        print(f"真实网络平均度：{un.average_degree(true_net)}".ljust(60), end=' ')
        print(f"构造网络平均度：{un.average_degree(model_net)}".ljust(60))
        top_nodes1 = {n for n, d in true_net.nodes(data=True) if d["bipartite"] == 0}
        top_nodes2 = {n for n, d in model_net.nodes(data=True) if d["bipartite"] == 0}
        print(f"真实网络密度：{bipartite.density(true_net, top_nodes1)}".ljust(60), end=' ')
        print(f"构造网络密度：{bipartite.density(model_net, top_nodes2)}".ljust(60))
        print(f"真实网络同配系数：{nx.degree_assortativity_coefficient(true_net)}".ljust(60), end=' ')
        print(f"构造网络同配系数：{nx.degree_assortativity_coefficient(model_net)}".ljust(60))
        print(f"真实网络平均聚类系数：{bipartite.average_clustering(true_net)}".ljust(60), end=' ')
        print(f"构造网络平均聚类系数：{bipartite.average_clustering(model_net)}".ljust(60))

        print('\n')

    folders = os.listdir(path4)
    folders.sort()
    # print(folders[:-1])
    for folder in folders[:-1]:
        print("真实网络")
        true_net = nx.read_gml(path1 + "21.gml")
        node_n = nx.number_of_nodes(true_net)
        edge_n = nx.number_of_edges(true_net)
        dia = un.diameter(true_net)
        ave_degree = un.average_degree(true_net)
        top_nodes = {n for n, d in true_net.nodes(data=True) if d["bipartite"] == 0}
        den = bipartite.density(true_net, top_nodes)
        degree_coef = nx.degree_assortativity_coefficient(true_net)
        clu = bipartite.average_clustering(true_net)
        print("节点数：%d   边数：%d   直径：%d   平均度：%f  密度：%f   同配系数：%f     平均聚类系数：%f" % (node_n, edge_n,
                                                                                  dia, ave_degree, den, degree_coef,
                                                                                  clu))
        files = os.listdir(path4 + folder)
        for file in files:
            print(re.sub("\\.gml", '', file))
            model_net = nx.read_gml(f"{path4}{folder}/{file}")
            node_n = nx.number_of_nodes(model_net)
            edge_n = nx.number_of_edges(model_net)
            dia = un.diameter(model_net)
            ave_degree = un.average_degree(model_net)
            top_nodes = {n for n, d in model_net.nodes(data=True) if d["bipartite"] == 0}
            den = bipartite.density(model_net, top_nodes)
            degree_coef = nx.degree_assortativity_coefficient(model_net)
            clu = bipartite.average_clustering(model_net)
            print("节点数：%d   边数：%d   直径：%d   平均度：%f  密度：%f   同配系数：%f     平均聚类系数：%f" % (node_n, edge_n,
                                                                                      dia, ave_degree, den, degree_coef,
                                                                                      clu))
        print('\n')
