import networkx as nx
import random
import numpy as np
import undirected_network as un
import os
import re


def create(g, nf0, ns0, t, m, n, l, name):
    """创建偏好连接选择模型"""
    g.add_nodes_from(list(range(nf0)), bipartite=0)   # 初始有NF0个基金节点，NS0个股票节点
    g.add_nodes_from(list(range(nf0, ns0+nf0)), bipartite=1)
    for node0 in range(nf0):
        for node1 in range(nf0, ns0+nf0):
            g.add_edge(node0, node1)

    for i in range(t):
        print(i)
        new_node = i + nf0 + ns0        # 在时刻t，加入一个新节点new_node，分为两种类型
        f_or_s = random.randint(0, 9)

        if f_or_s % 2 == 0:  # 基金节点
            g.add_node(new_node, bipartite=0)
            count = 0
            while count < m:
                bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
                in_degrees = [d for n, d in g.in_degree(bottom_nodes)]
                sum_degree = sum(in_degrees)
                for node in bottom_nodes:
                    decision = random.random()
                    pi = g.in_degree(node) / sum_degree
                    # print("*", pi, decision)
                    if decision <= pi:
                        if g.has_edge(new_node, node) is False:
                            g.add_edge(new_node, node)
                            # flag = 1
                            count += 1
                            break
        else:
            g.add_node(new_node, bipartite=1)   # 股票节点
            count = 0
            while count < n:
                top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
                old_node = random.choice(top_nodes)
                if g.has_edge(old_node, new_node) is False:
                    g.add_edge(old_node, new_node)
                    count += 1

        # 删边
        # p1 = random.random()
        # if p1 < p:
        count = 0
        while count < l:
            bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
            in_degrees = [d for n, d in g.in_degree(bottom_nodes)]
            sum_degree = sum(in_degrees)
            for s_node in bottom_nodes:
                decision = random.random()
                pi = g.in_degree(s_node) / sum_degree
                if decision < 1 - pi:
                    f_nodes = list(g.predecessors(s_node))
                    if len(f_nodes) != 0:
                        f_node = random.choice(f_nodes)
                        g.remove_edge(f_node, s_node)
                        count += 1
                        break
        # count = 0
        # while count < l:
        #     top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
        #     bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
        #     f_node = random.choice(top_nodes)
        #     s_node = random.choice(bottom_nodes)
        #     if g.has_edge(f_node, s_node) is True:
        #         g.remove_edge(f_node, s_node)
        #         count += 1

        # 随机连接
        # p2 = random.random()
        # if p2 < q:
        count = 0
        while count < l:
            top_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
            bottom_nodes = [n for n, d in g.nodes(data=True) if d["bipartite"] == 1]
            f_node = random.choice(top_nodes)
            in_degrees = [d for n, d in g.in_degree(bottom_nodes)]
            sum_degree = sum(in_degrees)
            # flag = 0
            # while flag == 0:
            # print(in_degrees)
            for s_node in bottom_nodes:
                decision = random.random()
                pi = g.in_degree(s_node) / sum_degree
                # print(pi, decision)
                if decision <= pi:
                    # print(g.has_edge(f_node, s_node))
                    if g.has_edge(f_node, s_node) is False:
                        g.add_edge(f_node, s_node)
                        count += 1
                        # flag = 1
                        break

    print(g)
    print(g.in_degree())
    print(g.out_degree())
    nx.write_gml(g, f'G:/jj_st/写论文/{name}.gml')


if __name__ == "__main__":
    # # nf0=3,ns0=5,t=1000,n=1,l=2 m变化
    # for _ in range(2, 6):
    #     G = nx.DiGraph()
    #     create(G, 3, 5, 1000, _, 1, 2, "m")
    # # nf0=3,ns0=5,m=3,n=1,l=2 t变化
    # for _ in [10000]:
    #     G = nx.DiGraph()
    #     create(G, 3, 5, _, 3, 1, 2, "t")
    # # nf0=3,ns0=5,t=1000,m=3,l=2 n变化
    # for _ in range(1, 4):
    #     G = nx.DiGraph()
    #     create(G, 3, 5, 1000, 3, _, 2,  "n")
    # # nf0=3,ns0=5,t=1000,m=3,n=1 l变化
    # for _ in range(1, 3):
    #     G = nx.DiGraph()
    #     create(G, 3, 5, 1000, 3, 1, _, "l")
    # # nf0=6,ns0=10,t=1000,m=5,q=0.1 p变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 1, _, 0.1, "p")
    # # nf0=6,ns0=10,t=1000,m=5,p=0.1 q变化
    # for _ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     G = nx.DiGraph()
    #     create(G, 6, 10, 1000, 1, 0.1, _, "q")
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

    # folders = os.listdir("G:/jj_st/偏好连接选择2/网络/")
    # for folder in folders:
    #     files = os.listdir(f"G:/jj_st/偏好连接选择2/网络/{folder}/")
    #     for file in files:
    #         graph = nx.read_gml(f"G:/jj_st/偏好连接选择2/网络/{folder}/{file}")
    #         pic_name = re.sub("\\.gml", '', file)
    #         un.di_degree_distribution(graph, f"G:/jj_st/偏好连接选择2/度分布/{folder}/", 0, {1: 0}, pic_name)

    G = nx.DiGraph()
    create(G, 2, 15, 5000, 10, 1, 2, "T=5000，N_F0=2，N_S0=15，m=10，n=1，l=2")
    # graph = nx.read_gml(f"G:/jj_st/偏好连接选择2/网络/t/t=12000.gml")
    # un.di_degree_distribution(graph, f"G:/jj_st/偏好连接选择2/度分布/t/", 0, {1: 0}, "t=12000")
