import pandas as pd
import networkx as nx
import os
import undirected_network as un
import re
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.algorithms.bipartite import centrality


def process_data(filename):
    """划分季度"""
    data = pd.read_csv(filename, usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
    # print(data.where(data['jjid'] == 0).dropna().sort_values(by='开始时间'))
    # print(data['开始时间'][0])
    date = data['开始时间'].drop_duplicates()
    date = list(date)
    date.sort()
    # print(date)
    date_to_quarter = dict(zip(date, range(len(date))))
    # print(date_to_quarter)
    data['季度'] = data['开始时间'].map(date_to_quarter)
    data = data.sort_values(by='季度')
    data.to_csv('jd.csv', index=False)


def process_data2(filename):
    """按季度分"""
    data = pd.read_csv(filename, low_memory=False)
    # print(data['jjid'].drop_duplicates(), data['stid'].drop_duplicates())
    quarters = list(data['季度'].drop_duplicates())
    # print(quarters)
    for quarter in quarters:
        frame = data.where(data['季度'] == quarter).dropna()
        frame['jjid'] = frame['jjid'].map(int)
        frame['stid'] = frame['stid'].map(int)
        frame['季度'] = frame['季度'].map(int)
        print(frame)
        frame.to_csv(path1 + f"{quarter}.csv", index=False)


'''def create_net(filename, path):
    g = nx.Graph()
    df = pd.read_csv(path + filename, low_memory=False)
    data = list(zip(list(df['jjid']), list(df['stid']), list(df['规模(万股)'])))
    # print(data)
    g.add_nodes_from(list(df['jjid']), bipartite=0)
    g.add_nodes_from(list(df['stid']), bipartite=1)
    g.add_weighted_edges_from(data)
    # print(g)
    # print(nx.is_bipartite(g))
    return g'''


def create_net(filename, path):
    g = nx.DiGraph()
    df = pd.read_csv(path + filename, low_memory=False)
    data = list(zip(list(df['jjid']), list(df['stid']), list(df['规模(万股)'])))
    # print(data)
    g.add_nodes_from(list(df['jjid']), bipartite=0)
    g.add_nodes_from(list(df['stid']), bipartite=1)
    g.add_weighted_edges_from(data)
    # print(g)
    # print(nx.is_bipartite(g))
    return g


def create_nets(folder_path1, folder_path2):
    files = os.listdir(folder_path1)
    files.sort()
    # print(files)
    for file in files:
        g = create_net(file, folder_path1)
        gml_name = re.sub('\\.csv', '', file)
        nx.write_gml(g, folder_path2 + f"{gml_name}.gml")


def diameters(path):
    """计算20个季度的直径并绘制折线图, 按无权图算的"""
    d_list = []
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        d = un.diameter(graph)
        # print(d)
        d_list.append(d)
    dtf = pd.DataFrame({'时间': rq, '直径': d_list})
    dtf.to_excel("G:/jj_st/bipartite/直径.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(d_list) + 1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络直径变化情况', fontsize='xx-large')
    plt.plot(x, d_list)
    plt.savefig('G:/基金-股票/直径.pdf', format='pdf', dpi=500)


def ave_degree(path):
    """计算20个季度的平均度并绘制折线图，按普通图计算的"""
    ave_out_degree_list = []
    ave_in_degree_list = []
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        # print(g)
        graph = nx.read_gml(path + g)
        a_o_d, a_i_d = un.average_degree(graph)
        # print(a_d)
        ave_out_degree_list.append(a_o_d)
        ave_in_degree_list.append(a_i_d)
    # a_s = pd.Series(ave_degree_list, index=rq)
    dtof = pd.DataFrame({'时间': rq, '出度平均度': ave_out_degree_list})
    dtof.to_excel("G:/jj_st/bipartite/出度平均度.xlsx", index=False)
    dtif = pd.DataFrame({'时间': rq, '入度平均度': ave_in_degree_list})
    dtif.to_excel("G:/jj_st/bipartite/入度平均度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_out_degree_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络出度平均度变化情况', fontsize='xx-large')
    plt.plot(x, ave_out_degree_list)
    # a_s.plot(rot='60')
    plt.savefig('G:/jj_st/bipartite/出度平均度.png', format='png', dpi=500)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_in_degree_list) + 1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络入度平均度变化情况', fontsize='xx-large')
    plt.plot(x, ave_in_degree_list)
    # a_s.plot(rot='60')
    plt.savefig('G:/jj_st/bipartite/入度平均度.png', format='png', dpi=500)


def degree_d(path, co_to_date):
    """计算80个季度的度分布并绘制散点图，按普通图计算"""
    gs = os.listdir(path)
    gs.sort()
    for g in gs[1:2]:
        print(g)
        graph = nx.read_gml(path + g)
        if len(graph.nodes()) > 0:
            un.di_degree_distribution(graph, 'G:/jj_st/one_mode_graph/jj/degree_distribution/digraph/0.2/',
                                       re.sub('\\.gml', '', g), co_to_date, '基金网络')


def assortativity(path):
    """计算20个季度的同配性并绘制折线图，按普通图计算"""
    r_list = []
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        r = nx.degree_assortativity_coefficient(graph)
        r_list.append(r)
    dtf = pd.DataFrame({'时间': rq, '同配系数': r_list})
    dtf.to_excel("G:/jj_st/bipartite/同配系数.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(r_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('基金-股票网络同配系数变化情况', fontsize='xx-large')
    plt.plot(x, r_list)
    # a_s.plot(rot='60')
    plt.savefig('G:/基金-股票/同配性.pdf', format='pdf', dpi=500)


def densities(path):
    """计算20个季度的密度并绘制折线图"""
    den_list = []
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        top_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
        den = bipartite.density(graph, top_nodes)
        # print(a_d)
        den_list.append(den)
    dtf = pd.DataFrame({'时间': rq, '密度': den_list})
    dtf.to_excel("G:/jj_st/bipartite/密度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(den_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络密度变化情况', fontsize='xx-large')
    plt.plot(x, den_list)
    # a_s.plot(rot='60')
    plt.savefig('G:/基金-股票/密度.pdf', format='pdf', dpi=500)


def convert(l):
    t = []
    colors = []
    for item in l:
        t.append((int(item[0]), int(item[1])))
        colors.append(int(item[0]))
    options = {
        "edgelist": t,
        "edge_color": colors,
        "width": 0.01,
        "edge_cmap": plt.cm.viridis,
    }
    return options


def convert_2(l):
    t = []
    for item in l:
        t.append((int(item[0]), int(item[1])))
    return t


def draw_net(path, filename, pos):
    """画图"""
    g = nx.read_gml(path + filename)
    plt.figure(figsize=(50, 30))
    nx.draw_networkx_nodes(g, pos, list(map(int, g.nodes())), node_size=1)
    options = convert(g.edges())
    nx.draw_networkx_edges(g, pos, **options)
    # print(convert(g.edges()))
    plt.axis('off')
    pic_name = re.sub('\\.gml', '', filename)
    title = number_to_rq[int(pic_name)]
    plt.title(title, fontsize=60)
    plt.savefig('G:/jj_st/nets/' + f"{pic_name}.png", format='png', dpi=70)


def draw_nets(path):
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        draw_net(path, g, position)


def diff_of_two_list(lst1, lst2):
    """实现lst1-lst2"""
    diff_list = []
    for item in lst1:
        if item not in lst2:
            diff_list.append(item)
    return diff_list


def draw_center_network(g, node_list, odd, net_name, savefig_path, co_to_date, pos=None):
    """画中心性图"""
    if pos is None:
        pos = nx.spring_layout(g)
    plt.figure(figsize=(50, 30))
    nx.draw_networkx_nodes(g, pos, list(map(int, odd)), node_size=1)
    nx.draw_networkx_nodes(g, pos, list(map(int, node_list)), node_size=3, node_color='red')
    # nx.draw_networkx_labels(g, pos, font_size=9)
    options = convert(g.edges())
    nx.draw_networkx_edges(g, pos, **options)
    plt.axis('off')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    title = co_to_date[int(net_name)]
    plt.title(title, fontsize=60)
    # plt.show()
    plt.savefig(savefig_path + f"{net_name}.png", format='png', dpi=70)


def degree_center(g, name, co_to_date, pos=None):
    """计算节点的度中心性"""
    top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
    node_degree_centrality = centrality.degree_centrality(g, top_nodes)
    degree_center_frame = pd.DataFrame(list(node_degree_centrality.items()), columns=('node', 'degree_centrality'))
    degree_center_frame = degree_center_frame.sort_values(by='degree_centrality', ascending=False)
    # print(degree_center_frame)
    degree_center_frame.to_csv('G:/jj_st/duzxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    top_10 = degree_center_frame.head(10)
    # top_10 = degree_center_frame.head(round(g.number_of_nodes()*0.05))
    all_node_list = list(degree_center_frame['node'])
    node_list = list(top_10['node'])
    # print("度中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/duzxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def betweenness_center(g, name, co_to_date, pos=None):
    """计算节点的介数中心性"""
    top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
    node_betweenness_centrality = centrality.betweenness_centrality(g, top_nodes)
    # print(node_betweenness_centrality)
    betweenness_centrality_frame = pd.DataFrame(list(node_betweenness_centrality.items()), columns=(
        'node', 'betweenness_centrality'))
    # print(node_betweenness_centrality_frame)
    betweenness_centrality_frame = betweenness_centrality_frame.sort_values(by='betweenness_centrality',
                                                                            ascending=False)
    betweenness_centrality_frame.to_csv('G:/jj_st/jszxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    top_10 = betweenness_centrality_frame.head(10)
    # top_10 = node_betweenness_centrality_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(betweenness_centrality_frame['node'])
    node_list = list(top_10['node'])
    # print("介数中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/jszxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def close_to_the_center(g, name, co_to_date, pos=None):
    """计算节点的接近中心性"""
    top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
    node_closeness_centrality = centrality.closeness_centrality(g, top_nodes)
    closeness_centrality_frame = pd.DataFrame(list(node_closeness_centrality.items()), columns=(
        'node', 'closeness_centrality'))
    closeness_centrality_frame = closeness_centrality_frame.sort_values(by='closeness_centrality', ascending=False)
    closeness_centrality_frame.to_csv('G:/jj_st/jjzxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    top_10 = closeness_centrality_frame.head(10)
    # top_10 = close_to_the_center_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(closeness_centrality_frame['node'])
    node_list = list(top_10['node'])
    # print("接近中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/jjzxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def center(path):
    """计算每张图的中心性"""
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        pic_name = re.sub('\\.gml', '', g)
        degree_center(graph, pic_name, number_to_rq, position)
        betweenness_center(graph, pic_name, number_to_rq, position)
        close_to_the_center(graph, pic_name, number_to_rq, position)


def max_matching(g, name, co_to_date, pos):
    """求网络的最大匹配"""
    top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
    match = bipartite.maximum_matching(g, top_nodes)
    match_frame = pd.DataFrame(list(match.items()), columns=('node1', 'node2'))
    match_frame.to_csv('G:/jj_st/maximum_matching_table/' + co_to_date[int(name)] + '.csv', index=False)
    end = int(len(list(match.items())) / 2)
    edges_set = list(match.items())[:end]
    remain_edges_set = diff_of_two_list(g.edges(), edges_set)
    # 绘图，最大匹配中的边为红色
    plt.figure(figsize=(50, 30))
    nx.draw_networkx_nodes(g, pos, list(map(int, g.nodes())), node_size=1)
    # nx.draw_networkx_labels(g, pos, font_size=9)
    nx.draw_networkx_edges(g, pos, convert_2(remain_edges_set), width=0.01)
    nx.draw_networkx_edges(g, pos, convert_2(edges_set), edge_color='red', width=0.02)
    plt.axis('off')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    title = co_to_date[int(name)]
    plt.title(title, fontsize=60)
    # plt.show()
    savefig_path = 'G:/jj_st/maximum_matching/'
    plt.savefig(savefig_path + f"{name}.png", format='png', dpi=70)


def matching(path):
    """求每张图的最大匹配"""
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        pic_name = re.sub('\\.gml', '', g)
        max_matching(graph, pic_name, number_to_rq, position)


def clustering(path):
    """计算20个季度的平均聚类系数并绘制折线图"""
    ave_cluster_list = []
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        ave_cluster = bipartite.average_clustering(graph)
        # print(a_d)
        ave_cluster_list.append(ave_cluster)
    dtf = pd.DataFrame({'时间': rq, '平均聚类系数': ave_cluster_list})
    dtf.to_excel("G:/jj_st/bipartite/平均聚类系数.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_cluster_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络平均聚类系数变化情况', fontsize='xx-large')
    plt.plot(x, ave_cluster_list)
    # a_s.plot(rot='60')
    plt.savefig('G:/jj_st/平均聚类系数.pdf', format='pdf', dpi=500)


def ave_in_degree_and_ave_out_degree(path):
    ave_in_degree = []
    ave_out_degree = []
    gs = os.listdir(path)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path + g)
        jj_number = len([n for n, d in graph.nodes(data=True) if d["bipartite"] == 0])
        st_number = len([n for n, d in graph.nodes(data=True) if d["bipartite"] == 1])
        sum_in_degree = sum([d for n, d in graph.in_degree()])
        sum_out_degree = sum([d for n, d in graph.out_degree()])
        ave_in_degree.append(sum_in_degree / st_number)
        ave_out_degree.append(sum_out_degree / jj_number)
    dtf0 = pd.DataFrame({'时间': rq, '股票平均入度': ave_in_degree})
    dtf0.to_excel("G:/jj_st/bipartite/股票平均入度.xlsx", index=False)
    dtf1 = pd.DataFrame({'时间': rq, '基金平均出度': ave_out_degree})
    dtf1.to_excel("G:/jj_st/bipartite/基金平均出度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_in_degree) + 1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络股票平均入度变化情况', fontsize='xx-large')
    plt.plot(x, ave_in_degree)
    # a_s.plot(rot='60')
    plt.savefig('G:/jj_st/bipartite/股票平均入度变化情况.png', format='png', dpi=500)

    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_out_degree) + 1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('基金-股票网络基金平均出度变化情况', fontsize='xx-large')
    plt.plot(x, ave_out_degree)
    # a_s.plot(rot='60')
    plt.savefig('G:/jj_st/bipartite/基金平均出度变化情况.png', format='png', dpi=500)


path1 = 'G:/jj_st/anjdfen/'
path2 = 'G:/jj_st/bipartite/anjdfen_graphs_gml/graph/'
path3 = 'G:/jj_st/bipartite/anjdfen_graphs_gml/digraph/'
# process_data('jjst.csv')
# process_data2('jd.csv')
# create_nets(path1, path2)
# print(graphs)
frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
rq = frame['开始时间'].drop_duplicates()
rq = list(rq)
rq.sort()
number_to_rq = dict(zip(list(range(len(rq))), rq))

df = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
jj = list(df['jjid'].drop_duplicates())
st = list(df['stid'].drop_duplicates())
x1 = [1 + 48 / len(jj) * i for i in range(len(jj))]
x2 = [1 + 48 / len(st) * i for i in range(len(st))]
y1 = [10] * len(jj)
y2 = [20] * len(st)
position1 = list(zip(x1, y1))
position2 = list(zip(x2, y2))
position = position1 + position2

# print(rq_labels)
# print(rq)
# print(rq)
# 直径
# diameters(path2)
# 平均度
# ave_degree(path2)
# 度分布
# degree_d('G:/jj_st/one_mode_graph/jj/jj_gml/digraph/0.2/', number_to_rq)
# 同配系数
# assortativity(path2)
# 密度
# densities(path2)
# 网络
# draw_nets(path2)
# 中心性
# center(path2)
# 最大匹配
# matching(path2)
# 平均聚类系数
# clustering(path2)
# 平均最短路径长度
# average_shortest_path(path2)
# 平均出入度
ave_in_degree_and_ave_out_degree(path3)
'''b = nx.Graph()
b.add_nodes_from([1, 2, 3], bipartite=0)
b.add_nodes_from([4, 5, 6], bipartite=1)
b.add_edges_from([(1, 4), (1, 5), (2, 4), (2, 5), (2, 6), (3, 6)])
degree_center(b, "00", number_to_rq, position)'''
