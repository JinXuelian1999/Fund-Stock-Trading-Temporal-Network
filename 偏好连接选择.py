import networkx as nx
import random
import numpy as np
from scipy import stats
import undirected_network as un
import os
import re


def create(g, nf0, ns0, t, p1, p2, p3, p4, p5, p6, p, m, name):
    """创建偏好连接选择模型"""
    # g = nx.DiGraph()
    # print("*")
    g.add_nodes_from(list(range(nf0)), bipartite=0, delta=0)        # 初始有NF0个基金节点，NS0个股票节点
    g.add_nodes_from(list(range(nf0, ns0+nf0)), bipartite=1, decrement=0)
    for node0 in range(nf0):
        for node1 in range(nf0, ns0+nf0):
            g.add_edge(node0, node1)
            g.nodes(data=True)[node0]['delta'] += 1
            g.nodes(data=True)[node1]['decrement'] -= 1

    for i in range(t):
        # print(i)
        new_node = i + nf0 + ns0     # 在时刻t，加入一个新节点new_node，分为两种类型
        f_or_s = random.randint(0, 9)

        if f_or_s % 2 == 0:        # 基金节点
            g.add_node(new_node, bipartite=0, delta=0)
            count = 0
            while count < m:
                p11 = random.random()
                if p11 < p1:      # 以概率p1随机选择而一个股票节点进行连接
                    bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
                    old_node = random.choice(bottom_nodes)
                    if g.has_edge(new_node, old_node) is False:
                        g.add_edge(new_node, old_node)
                        count += 1
                        g.nodes(data=True)[new_node]['delta'] += 1
                        g.nodes(data=True)[old_node]['decrement'] -= 1
                else:       # 以概率（1-p1）正比于节点入度，随机选择一个股票节点连接
                    bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
                    in_degrees = [d for n, d in g.in_degree(bottom_nodes)]
                    sum_degree = sum(in_degrees)
                    flag = 0
                    while flag == 0:
                        for node in bottom_nodes:
                            decision = random.random()
                            pi = g.in_degree(node) / sum_degree
                            if decision < pi:
                                if g.has_edge(new_node, node) is False:
                                    g.add_edge(new_node, node)
                                    flag = 1
                                    count += 1
                                    g.nodes(data=True)[new_node]['delta'] += 1
                                    g.nodes(data=True)[node]['decrement'] -= 1
                                    break
        else:
            g.add_node(new_node, bipartite=1, decrement=0)       # 股票节点
            p22 = random.random()
            if p22 < p2:          # 以概率p2随机选择一个基金节点，进行连接
                top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
                old_node = random.choice(top_nodes)
                if g.has_edge(old_node, new_node) is False:
                    g.add_edge(old_node, new_node)
                    g.nodes(data=True)[old_node]['delta'] += 1
                    g.nodes(data=True)[new_node]['decrement'] -= 1
            else:               # 以概率1-p2，基金节点出度服从泊松分布，随机选择一个基金节点
                top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
                out_degrees = [d for n, d in g.out_degree(top_nodes)]
                mean_degree = np.mean(out_degrees)
                flag = 0
                while flag == 0:
                    for node in top_nodes:
                        decision = random.random()
                        pi = stats.poisson.pmf(g.out_degree(node)+1, mean_degree)
                        if decision < pi:
                            if g.has_edge(node, new_node) is False:
                                g.add_edge(node, new_node)
                                flag = 1
                                g.nodes(data=True)[node]['delta'] += 1
                                g.nodes(data=True)[new_node]['decrement'] -= 1
                                break

        p33 = random.random()       # 随机连接
        if p33 < p3:        # 以概率p3，随机选取起点与终点，增加一条边。
            top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
            bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
            f_node = random.choice(top_nodes)
            s_node = random.choice(bottom_nodes)
            g.add_edge(f_node, s_node)
            g.nodes(data=True)[f_node]['delta'] += 1
            g.nodes(data=True)[s_node]['decrement'] -= 1
        else:       # 以概率1-p3，边的起始点随机选取，终点则以概率(〖IX〗_j (t))/(∑_k▒〖〖IX〗_k (t)〗)偏好连接。
            top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
            bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
            f_node = random.choice(top_nodes)
            in_degrees = [d for n, d in g.in_degree(bottom_nodes)]
            sum_degree = sum(in_degrees)
            flag = 0
            while flag == 0:
                for s_node in bottom_nodes:
                    decision = random.random()
                    pi = g.in_degree(s_node) / sum_degree
                    if decision < pi:
                        g.add_edge(f_node, s_node)
                        flag = 1
                        g.nodes(data=True)[f_node]['delta'] += 1
                        g.nodes(data=True)[s_node]['decrement'] -= 1
                        break

        p_ = random.random()
        if p_ < p:
            p44 = random.random()       # 删边（交易），暂时不考虑增持/减持（标量变化）
            if p44 < p4:    # 以概率p4，随机选择一个股票节点，删边
                bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
                s_node = random.choice(bottom_nodes)
                f_nodes = list(g.predecessors(s_node))
                if len(f_nodes) != 0:
                    f_node = random.choice(f_nodes)
                    g.remove_edge(f_node, s_node)
                    g.nodes(data=True)[f_node]['delta'] += 1
                    g.nodes(data=True)[s_node]['decrement'] += 1
            else:       # 以概率1-p4，以概率(1-(〖IX〗_i (t))/(∑_j▒〖〖IX〗_j (t)〗))（S节点入度越大，被选中的概率越小），选择一个股票节点，删边
                bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
                in_degrees = [d for n, d in g.in_degree(bottom_nodes)]
                sum_degree = sum(in_degrees)
                flag = 0
                while flag == 0:
                    for s_node in bottom_nodes:
                        decision = random.random()
                        pi = g.in_degree(s_node) / sum_degree
                        if decision < 1-pi:
                            f_nodes = list(g.predecessors(s_node))
                            if len(f_nodes) != 0:
                                f_node = random.choice(f_nodes)
                                g.remove_edge(f_node, s_node)
                                flag = 1
                                g.nodes(data=True)[f_node]['delta'] += 1
                                g.nodes(data=True)[s_node]['decrement'] += 1
                                break

        # p__ = random.random()
        # if p__ < p:
        #     f_or_s = random.randint(0, 9)       # 节点的退出机制
        #     if f_or_s % 2 == 0:        # 基金节点
        #         p55 = random.random()
        #         if p55 < p5:
        #             top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
        #             f_node = random.choice(top_nodes)
        #             f_node_suc = list(g.successors(f_node))
        #             if len(f_node_suc) != 0:
        #                 for node in f_node_suc:
        #                     g.nodes(data=True)[node]['decrement'] += 1
        #             g.remove_node(f_node)
        #         else:
        #             out_degree_delta = [d["delta"] for n, d in g.nodes(data=True) if d["bipartite"] == 0]
        #             sum_delta = sum(out_degree_delta)
        #             top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
        #             flag = 0
        #             while flag == 0:
        #                 for f_node in top_nodes:
        #                     decision = random.random()
        #                     pi = g.nodes(data=True)[f_node]['delta'] / sum_delta
        #                     if decision < pi:
        #                         f_node_suc = list(g.successors(f_node))
        #                         if len(f_node_suc) != 0:
        #                             for node in f_node_suc:
        #                                 g.nodes(data=True)[node]['decrement'] += 1
        #                         g.remove_node(f_node)
        #                         flag = 1
        #                         break
        #     else:    # 股票节点
        #         p66 = random.random()
        #         if p66 < p6:
        #             bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
        #             s_node = random.choice(bottom_nodes)
        #             s_node_pre = list(g.predecessors(s_node))
        #             if len(s_node_pre) != 0:
        #                 for node in s_node_pre:
        #                     g.nodes(data=True)[node]['delta'] += 1
        #             g.remove_node(s_node)
        #         else:
        #             in_degree_decrement = [d["decrement"] for n, d in g.nodes(data=True) if d["bipartite"] == 1]
        #             sum_decrement = sum(in_degree_decrement)
        #             bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
        #             for s_node in bottom_nodes:
        #                 decision = random.random()
        #                 pi = g.nodes(data=True)[s_node]['decrement'] / sum_decrement
        #                 if decision < pi:
        #                     s_node_pre = list(g.predecessors(s_node))
        #                     if len(s_node_pre) != 0:
        #                         for node in s_node_pre:
        #                             g.nodes(data=True)[node]['delta'] += 1
        #                     g.remove_node(s_node)
        #                     break
    print(g)
    print(g.in_degree())
    print(g.out_degree())
    if not os.path.exists(f'G:/jj_st/偏好连接选择/网络/{name}/'):
        os.mkdir(f'G:/jj_st/偏好连接选择/网络/{name}/')
    nx.write_gml(g, f'G:/jj_st/偏好连接选择/网络/{name}/{name}={eval(name)}.gml')


if __name__ == "__main__":
    # nf0=6,ns0=10,t=1000,p1=p2=p3=p4=p5=p6=p=0.1 m变化
    # for _ in range(2, 10):
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, _, "m")
    # # nf0=6,ns0=10,p1=p2=p3=p4=p5=p6=p=0.1,m=5,t变化
    # for _ in [100, 1000, 10000]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, _, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5, "t")
    # # nf0=6,ns0=10,t=1000,p2=p3=p4=p5=p6=p=0.1,m=5 p1变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, _, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5, "p1")
    # # nf0=6,ns0=10,t=1000,p1=p3=p4=p5=p6=p=0.1,m=5 p2变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, _, 0.1, 0.1, 0.1, 0.1, 0.1, 5, "p2")
    # # nf0=6,ns0=10,t=1000,p1=p2=p4=p5=p6=p=0.1,m=5 p3变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, 0.1, _, 0.1, 0.1, 0.1, 0.1, 5, "p3")
    # # nf0=6,ns0=10,t=1000,p1=p2=p3=p5=p6=p=0.1,m=5 p4变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, 0.1, 0.1, _, 0.1, 0.1, 0.1, 5, "p4")
    # # nf0=6,ns0=10,t=1000,p1=p2=p3=p4=p6=p=0.1,m=5 p5变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, 0.1, 0.1, 0.1, _, 0.1, 0.1, 5, "p5")
    # # nf0=6,ns0=10,t=1000,p1=p2=p3=p4=p5=p=0.1,m=5 p6变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, 0.1, 0.1, 0.1, 0.1, _, 0.1, 5, "p6")
    # # nf0=6,ns0=10,t=1000,p1=p2=p3=p4=p5=p6=0.1,m=5 p变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, _, 5, "p")

    # folders = os.listdir("G:/jj_st/偏好连接选择/网络/")
    # for folder in folders:
    #     files = os.listdir(f"G:/jj_st/偏好连接选择/网络/{folder}/")
    #     for file in files:
    #         graph = nx.read_gml(f"G:/jj_st/偏好连接选择/网络/{folder}/{file}")
    #         pic_name = re.sub("\\.gml", '', file)
    #         un.di_degree_distribution(graph, f"G:/jj_st/偏好连接选择/度分布/{folder}/", 0, {1: 0}, pic_name)
    G = nx.DiGraph()
    create(G, 2, 15, 5000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, _, "m")





