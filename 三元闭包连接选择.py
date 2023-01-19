import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os


def draw_ternary(pb, pr, name, signal):
    """画图"""
    plt.figure(figsize=(22, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    x = list(range(11))

    plt.plot(x, pb, color='red', linestyle='--', label='Pb(k)')  # 画出p(k)作为k的一个函数，表示共同基金的数量对建立连接的影响。
    plt.legend()
    pb.pop()
    plt.plot(list(range(1, 11)), pb, color='red', linestyle='--')

    plt.plot(x, pr, label='P(k)')  # 理论曲线Pb(k) = 1-(1-p)^k
    plt.legend()

    plt.xlabel('共同的基金数k', fontsize='x-large')
    plt.ylabel('建立连接的几率', fontsize='x-large')
    # print(name)
    pic_name = number_to_rq[name] + '-' + number_to_rq[name+1]
    if signal == 0:
        plt.savefig(f'G:/jj_st/三元闭包选择连接/p0/{pic_name}.png', format='png', dpi=500)
    else:
        plt.savefig(f'G:/jj_st/三元闭包选择连接/p1/{pic_name}.png', format='png', dpi=500)
    # plt.show()


def ternary(net0, net1, name):
    """三元闭包连接选择模型"""
    pr = []

    for k in range(11):
        # print(k)
        node_pairs = []
        n = 0
        for node0 in net0.nodes():
            for node1 in net0.nodes():
                if node0 != node1:
                    node0_neighbors = net0.neighbors(node0)
                    node1_neighbors = net0.neighbors(node1)
                    len_isn = len(list(set(node0_neighbors).intersection(set(node1_neighbors))))        # 对于每个k，在第1次快照
                    # 中找出恰好有k个共同基金的节点对，且它们之间没有边。
                    if len_isn == k and net0.has_edge(node0, node1) is False:
                        node_pairs.append((node0, node1))
        for pair in node_pairs:
            if net1.has_edge(*pair) is True:
                n += 1
        p = n/len(node_pairs)       # 定义p(k)为这些对节点在第2次快照中形成了连接的比例，这就是在两个有k个共同基金的节点之间建立连接的几率。
        pr.append(p)

    print(pr)
    pb0 = [1-(1-pr[0])**k for k in range(11)]
    print(pb0)
    pb1 = [1-(1-pr[1])**k for k in range(11)]
    print(pb1)
    draw_ternary(pb0, pr, name, 0)
    draw_ternary(pb1, pr, name, 1)


frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
rq = frame['开始时间'].drop_duplicates()
rq = list(rq)
rq.sort()
number_to_rq = dict(zip(list(range(len(rq))), rq))
path = 'G:/jj_st/one_mode_graph/jj/jj_gml/graph/0.1/'

gs = os.listdir(path)
gs.sort()

for q in range(21, 79):
    print(q)
    network0 = nx.read_gml(path + gs[q])        # 选取连续2个时间点的网络快照
    network1 = nx.read_gml(path + gs[q+1])
    ternary(network0, network1, q)
