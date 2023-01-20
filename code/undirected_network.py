import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from networkx.algorithms import community
from networkx.algorithms import centrality
from networkx.algorithms import core
import itertools
from networkx.algorithms.assortativity import degree_assortativity_coefficient
from networkx.algorithms.assortativity import average_degree_connectivity
from itertools import combinations
from collections import Counter
from networkx.algorithms.isomorphism import is_isomorphic
import matplotlib
import os


def char_to_code(filename1, columns, filename2):
    """给表中列编码"""
    df = pd.read_excel(filename1)
    df = df.where(df.notnull(), np.nan)  # None的地方赋值为nan
    # print(df)
    list1 = []
    for column in columns:
        list1 = list1 + list(df[column].dropna().unique())
        # print(list1)
    list2 = list(set(list1))
    list2.sort(key=list1.index)
    ch_to_co = dict(zip(list2, range(len(list2))))

    # co_to_ch = dict(zip(range(len(list2)), list2))
    # print(ch_to_co)
    ch_to_co[np.nan] = np.nan
    for column in columns:
        df[column] = df[column].map(ch_to_co)
    df.to_excel(filename2, index=False)
    return len(list2)


def create_graph(filename1, filename2):
    """构建社交网络"""
    G = nx.Graph()
    names_number = char_to_code(filename1, ['姓名', '与谁是朋友'], filename2)
    df = pd.read_excel(filename2)
    data1 = np.array(df[['姓名', '年龄', '性别', '家乡（省，直辖市）', '学历', '行业', '性格',
                         '当前工作/上学地点（省，直辖市）']].drop_duplicates())
    for i in range(len(data1[:, 0])):
        G.add_node(int(data1[i][0]), age=data1[i][1], sex=data1[i][2], place1=data1[i][3], edu=data1[i][4],
                   job=data1[i][5], personality=data1[i][6], place2=data1[i][7])

    data2 = np.array(df[['姓名', '与谁是朋友']].dropna(how='any'))
    for i in range(len(data2[:, 0])):
        G.add_edge(data2[i][0], data2[i][1])

    '''pos = nx.random_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='r', alpha=1)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.7, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=9)'''
    plt.figure(figsize=(30, 30))
    nx.draw_networkx(G, with_labels=True, node_size=100, node_color='r', width=1, edge_color='black',
                     font_size=9)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('国研211计算机与电子信息学生社交网络')
    plt.axis('off')
    plt.show()
    return G


def diameter(g):
    """计算无向图的直径"""
    path_length = 0
    # path = []
    for node in g.nodes():
        pa_len = nx.single_source_shortest_path(g, node)
        # print(pa_len)
        for key in pa_len.keys():
            if len(pa_len[key]) > path_length:
                path_length = len(pa_len[key])
                # path = pa_len[key]
    path_length -= 1
    # print("无向网络的直径的路径之一为：", list(pd.Series(path).map(int)))
    # print("无向网络的直径为：", path_length)
    return path_length


def average_degree(g):
    """计算平均出度"""
    sum_degree = 0
    for v in g.nodes():
        sum_degree += g.degree(v)
    nodes_number = len(g.nodes())
    if nodes_number != 0:
        ave_degree = sum_degree/nodes_number
    else:
        ave_degree = np.nan
    return ave_degree


def clustering_coefficient(g):
    """计算节点聚类系数分布、平均以及全局聚类系数"""
    arr = np.zeros((len(g.nodes()), len(g.nodes())), dtype=np.float64)
    for i in range(len(g.nodes())):
        for j in range(i+1, len(g.nodes())):
            if nx.has_path(g, i, j) is True and nx.shortest_path_length(g, source=i, target=j) == 1:
                arr[i][j] = 1
                arr[j][i] = 1
    arr2 = np.dot(arr, arr)
    arr3 = np.dot(arr2, arr)
    arr4 = []
    s = 0
    num_total_triplets = 0
    for i in range(len(g.nodes())):
        if arr2[i][i] * (arr2[i][i] - 1) != 0:
            arr4.append(arr3[i][i] / (arr2[i][i] * (arr2[i][i] - 1)))
            num_total_triplets += (arr2[i][i] * (arr2[i][i] - 1))/2
            print(f"节点{i}的聚类系数为：{arr4[i]}")
            s += arr4[i]
        else:
            print(f"节点{i}的聚类系数为：0.0")
            arr4.append(0.0)
    # print('无向网络平均聚类系数为：', s/len(g.nodes()))
    num_closed_triplets = sum(np.diag(arr3))/2
    # print('无向网络全局聚类系数为：', num_closed_triplets / num_total_triplets)
    return num_closed_triplets / num_total_triplets
    # plt.scatter(sorted(g.nodes()), arr4, color='purple', s=10)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.title('无向网络聚类系数分布')
    # plt.xlabel('nodes')
    # plt.ylabel('clustering_coefficient')
    # plt.show()
    # print(nx.clustering(g))
    # print(nx.average_clustering(g))
    # print(nx.transitivity(g))


def in_degree_histogram(graph):
    counts = Counter(d for n, d in graph.in_degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1)]


def out_degree_histogram(graph):
    counts = Counter(d for n, d in graph.out_degree())
    return [counts.get(i, 0) for i in range(max(counts) + 1)]


def draw_degree_distribution(p, savefig_path, count, co_to_date, name):
    """画度分布图"""
    x = list(range(len(p)))
    arr = np.array([x, p]).T
    d_d = pd.DataFrame(arr, columns=['k', 'P(k)'])
    plt.figure(figsize=(10, 6))
    d_d.plot('k', 'P(k)', kind='scatter')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.title(f'{co_to_date[int(count)]} ' + name, fontsize='xx-large')
    # plt.savefig(savefig_path + f"{count}.png", format='png', dpi=100)
    plt.title(name, fontsize='xx-large')
    os.makedirs(savefig_path, exist_ok=True)
    plt.savefig(savefig_path + f"{name}.png", format='png', dpi=500)
    plt.close()


def draw_degree_distribution_log(p, savefig_path, count, co_to_date, name):
    """画单log度分布图"""
    x = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(p))]
    arr = np.array([p, x]).T
    d_d = pd.DataFrame(arr, columns=['P(k)', 'k'])
    plt.figure(figsize=(10, 6))
    d_d.plot('k', 'P(k)', kind='scatter')
    plt.xscale('log')
    # plt.yscale('log')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.title(f'{co_to_date[int(count)]} ' + name, fontsize='xx-large')
    # plt.savefig(savefig_path + f"{count}.png", format='png', dpi=100)
    plt.title(name, fontsize='xx-large')
    os.makedirs(savefig_path, exist_ok=True)
    plt.savefig(savefig_path + f"{name}.png", format='png', dpi=500)
    plt.close()


def draw_degree_distribution_2log(x, y, savefig_path, count, co_to_date, name):
    """画双log度分布图"""
    arr = np.array([x, y]).T
    d_d = pd.DataFrame(arr, columns=['k', 'P(k)'])
    plt.figure(figsize=(10, 6))
    d_d.plot('k', 'P(k)', kind='scatter')
    plt.xscale('log')
    plt.yscale('log')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.title(f'{co_to_date[int(count)]} ' + name, fontsize='xx-large')
    # plt.savefig(savefig_path + f"{count}.png", format='png', dpi=100)
    plt.title(name, fontsize='xx-large')
    os.makedirs(savefig_path, exist_ok=True)
    plt.savefig(savefig_path + f"{name}.png", format='png', dpi=500)
    plt.close()


def degree_distribution(g, savefig_path, count, co_to_date, name):
    """计算度分布"""
    counts = nx.degree_histogram(g)
    x = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(counts))]
    p = [i / sum(counts) for i in counts]
    y = [pow(10, np.log10(i)) if i > 0 else 0 for i in p]
    draw_degree_distribution(p, savefig_path + '普通/', count, co_to_date, name)
    draw_degree_distribution_log(p, savefig_path, count, co_to_date, name)
    draw_degree_distribution_2log(x, y, savefig_path + '双log/', count, co_to_date, name)


def di_degree_distribution(g, savefig_path, count, co_to_date, name):
    """计算有向图的度分布"""
    in_counts = in_degree_histogram(g)
    out_counts = out_degree_histogram(g)

    in_p = [i / sum(in_counts) for i in in_counts]
    out_p = [i / sum(out_counts) for i in out_counts]

    x1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(in_counts))]
    y1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in in_p]

    x2 = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(out_counts))]
    y2 = [pow(10, np.log10(i)) if i > 0 else 0 for i in out_p]
    print(x2)
    print(y2)

    draw_degree_distribution(in_p, savefig_path + '普通/入度分布/', count, co_to_date, name + "入度分布")
    draw_degree_distribution(out_p, savefig_path + '普通/出度分布/', count, co_to_date, name + "出度分布")
    draw_degree_distribution_log(in_p, savefig_path + '单log/入度分布/', count, co_to_date, name + "入度分布")
    draw_degree_distribution_log(out_p, savefig_path + '单log/出度分布/', count, co_to_date, name + "出度分布")
    draw_degree_distribution_2log(x1, y1, savefig_path + '双log/入度分布/', count, co_to_date, name + "入度分布")
    draw_degree_distribution_2log(x2, y2, savefig_path + '双log/出度分布/', count, co_to_date, name + "出度分布")


def assortativity1(g):
    """同配系数分析同配性"""
    degree_list = []
    for item in g.degree():
        degree_list.append(item[1])
    degree_list = sorted(list(set(degree_list)))
    part1 = 0
    part2 = 0
    part3 = 0
    n = max(degree_list) + 1
    arr = np.zeros((n, n), dtype=np.int32)
    e = np.zeros((n, n), dtype=np.float64)
    q = []
    for edge in g.edges():
        arr[g.degree(edge[0])][g.degree(edge[1])] += 1
    # print(arr)
    for de1 in range(n):
        for de2 in range(de1, n):
            if de1 != de2:
                e[de1][de2] = (arr[de1][de2] + arr[de2][de1]) / (2 * g.number_of_edges())
                e[de2][de1] = e[de1][de2]
            else:
                e[de1][de2] = arr[de1][de2] / g.number_of_edges()
    # print(e)
    for de in range(n):
        q.append(sum(e[de][:]))
    # print(q)
    for d in degree_list:
        part1 += d ** 2 * q[d]
        part2 += d * q[d]
    delta_square = part1 - part2 ** 2
    # print(part1, part2, delta_square)
    for j in range(n):
        for k in range(n):
            part3 += j * k * (e[j][k] - q[j] * q[k])
    # print(part3)
    r = part3 / delta_square
    # print(q)
    # print(part3, delta_square)
    return r


def assortativity2(g):
    """Knn分析同配性"""
    arr = np.zeros((len(g.nodes()), len(g.nodes())), dtype=np.float64)
    for i in list(g.nodes()):
        for j in range(i + 1, len(g.nodes())):
            if nx.has_path(g, i, j) is True and nx.shortest_path_length(g, source=i, target=j) == 1:
                arr[i][j] = 1
                arr[j][i] = 1
    list1 = []
    list2 = []
    for i in list(g.nodes()):
        degree_sum = 0
        for j in list(g.nodes()):
            degree_sum += arr[i][j] * g.degree(j)
        list1.append(degree_sum / g.degree(i))
        list2.append(g.degree(i))
    df = pd.DataFrame({'de': list2, 'knni': list1})
    grouped = df['knni'].groupby(df['de'])
    knn = grouped.mean()

    # print(knn)
    plt.plot(list(knn.index), list(knn.values), alpha=0.7, label='Knn(k)', marker='o')
    # plt.xlim(0.5, 20.5)
    # plt.ylim(0.5, 11.5)
    x_major_locator = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid()
    plt.legend()
    plt.show()


def random_color():
    """随机生成颜色"""
    color_arr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += color_arr[random.randint(0, 15)]
    return "#"+color


def cnm(g, pos=None):
    """利用CNM算法划分社团"""
    communities = community.greedy_modularity_communities(g)
    print(communities)
    mod = community.modularity(g, communities)
    size = int(len(communities))
    print("无向网络CNM划分的社团个数：", size)
    print("无向网络CNM模块度：", mod)
    print("无向网络CNM分区覆盖率：", community.coverage(g, communities))
    if pos is None:
        pos = nx.spring_layout(g)
    cnt = 0
    for com in communities:
        cnt += 1
        list_nodes = list(com)
        nx.draw_networkx_nodes(g, pos, list_nodes, node_size=20, node_color=random_color(), label=f"community{cnt}")
    nx.draw_networkx_edges(g, pos, edge_color='gray', alpha=0.7)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('无向网络利用CNM算法划分社团')
    plt.legend()
    plt.axis('off')

    plt.show()


'''def k_clique(g):
    """寻找K-派系社团"""
    communities = community.k_clique_communities(g, 3)
    # print("生成的社区数：%d" % len(communities))
    # print(list(communities[0]))
    list1 = []
    list2 = []
    limited = itertools.takewhile(lambda x: len(x) <= g.number_of_nodes(), communities)
    for com in limited:
        list1.append(com)
    print(list1)
    mod = community.modularity(g, communities)
    size = float(len(communities))
    print("划分的社团个数：", size)
    print("K派系社团模块度：", mod)
    pos = nx.spring_layout(g)
    for com in communities:
        list_nodes = list(com)
        nx.draw_networkx_nodes(g, pos, list_nodes, node_size=20, node_color=random_color())
    nx.draw_networkx_edges(g, pos, edge_color='gray', alpha=0.7)
    plt.axis('off')
    plt.show()'''
    

def gn(g, pos=None):
    """G-N算法"""
    list1 = []
    list2 = []
    comp = community.girvan_newman(g)
    limited = itertools.takewhile(lambda x: len(x) <= g.number_of_nodes(), comp)
    for com in limited:
        list1.append(com)
        list2.append(community.modularity(g, com))
    in_x = list2.index(max(list2))
    communities = list1[in_x]
    print(communities)
    mod = list2[in_x]
    size = len(communities)
    print("无向网络G-N法划分社团个数：%d" % size)
    print("无向网络G-N法模块度：%f" % mod)
    print("无向网络G-N法分区覆盖率：%f" % community.coverage(g, communities))
    if pos is None:
        pos = nx.spring_layout(g)
    cnt = 0
    for c in communities:
        cnt += 1
        list_nodes = list(c)
        nx.draw_networkx_nodes(g, pos, list_nodes, node_size=20, node_color=random_color(), label=f"community{cnt}")
    nx.draw_networkx_edges(g, pos, edge_color='gray', alpha=0.7)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('无向网络利用G-N算法划分社团')
    plt.legend()
    plt.axis('off')
    plt.show()


def density(g):
    """计算网络的密度"""
    d = g.number_of_edges() / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
    return d


def diff_of_two_list(lst1, lst2):
    """实现lst1-lst2"""
    diff_list = []
    for item in lst1:
        if item not in lst2:
            diff_list.append(item)
    return diff_list


def convert(l):
    t = []
    for item in l:
        t.append((int(item[0]), int(item[1])))
    return t


def draw_center_network(g, node_list, odd, net_name, savefig_path, co_to_date, pos=None):
    """画中心性图"""
    if pos is None:
        pos = nx.spring_layout(g)
    plt.figure(figsize=(50, 30))
    nx.draw_networkx_nodes(g, pos, list(map(int, odd)), node_size=2)
    nx.draw_networkx_nodes(g, pos, list(map(int, node_list)), node_size=3, node_color='red')
    # nx.draw_networkx_labels(g, pos, font_size=9)
    nx.draw_networkx_edges(g, pos, convert(g.edges()), width=0.02)
    plt.axis('off')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    title = co_to_date[int(net_name)]
    plt.title(title, fontsize=60)

    plt.savefig(savefig_path + f"{net_name}.png", format='png', dpi=70)


def degree_center(g, name, co_to_date, pos=None):
    """计算节点的度中心性"""
    degree_center_frame = pd.DataFrame(columns=('node', 'degree_centrality'))
    for node in g.nodes():
        node_degree_center = g.degree(node) / (g.number_of_nodes() - 1)
        degree_center_frame = degree_center_frame.append([{'node': node, 'degree_centrality': node_degree_center}],
                                                         ignore_index=True)
        # print("节点%d的度中心性为%f。" % (node, node_degree_center))
    degree_center_frame = degree_center_frame.sort_values(by='degree_centrality', ascending=False)
    # print(degree_center_frame)
    degree_center_frame.to_csv('G:/jj_st/duzxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    # top_10 = degree_center_frame.head(10)
    top_10 = degree_center_frame.head(round(g.number_of_nodes()*0.05))
    all_node_list = list(degree_center_frame['node'])
    node_list = list(top_10['node'])
    print("度中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/duzxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def betweenness_center(g, name, co_to_date, pos=None):
    """计算节点的介数中心性"""
    node_betweenness_centrality = centrality.betweenness_centrality(g)
    # print(node_betweenness_centrality)
    node_betweenness_centrality_frame = pd.DataFrame(list(node_betweenness_centrality.items()),
                                                     columns=('node', 'betweenness_centrality'))
    # print(node_betweenness_centrality_frame)
    node_betweenness_centrality_frame = node_betweenness_centrality_frame.sort_values(by='betweenness_centrality',
                                                                                      ascending=False)
    node_betweenness_centrality_frame.to_csv('G:/jj_st/jszxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    # top_10 = node_betweenness_centrality_frame.head(10)
    top_10 = node_betweenness_centrality_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(node_betweenness_centrality_frame['node'])
    node_list = list(top_10['node'])
    print("介数中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/jszxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def close_to_the_center(g, name, co_to_date, pos=None):
    """计算节点的接近中心性"""
    close_to_the_center_frame = pd.DataFrame(columns=('node', 'close_to_the_center'))
    for node in g.nodes():
        d = sum(nx.shortest_path_length(g, source=node).values()) / (g.number_of_nodes() - 1)
        cc = 1 / d
        close_to_the_center_frame = close_to_the_center_frame.append([{'node': node,
                                                                       'close_to_the_center': cc}],
                                                                     ignore_index=True)
    close_to_the_center_frame = close_to_the_center_frame.sort_values(by='close_to_the_center', ascending=False)
    close_to_the_center_frame.to_csv('G:/jj_st/jjzxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    # top_10 = close_to_the_center_frame.head(10)
    top_10 = close_to_the_center_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(close_to_the_center_frame['node'])
    node_list = list(top_10['node'])
    print("接近中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/jjzxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def e_center(g, name, co_to_date, pos=None):
    """特征向量中心性"""
    eig_cen = centrality.eigenvector_centrality_numpy(g)
    eigenvector_centrality_frame = pd.DataFrame(list(eig_cen.items()), columns=('node', 'eigenvector_centrality'))
    eigenvector_centrality_frame = eigenvector_centrality_frame.sort_values(by='eigenvector_centrality', ascending=False)
    eigenvector_centrality_frame.to_csv('G:/jj_st/tzxlzxx_table/' + co_to_date[int(name)] + '.csv', index=False)
    # top_10 = eigenvector_centrality_frame.head(10)
    top_10 = eigenvector_centrality_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(eigenvector_centrality_frame['node'])
    node_list = list(top_10['node'])
    print("特征向量中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    path = 'G:/jj_st/tzxlzxx/'
    draw_center_network(g, node_list, odd, name, path, co_to_date, pos)


def draw_k_shell(g, pos=None):
    """k-壳分解"""
    k = 1
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    g1 = core.k_shell(g, k)
    if pos is None:
        pos = nx.spring_layout(g)
    while len(g1.nodes()) != 0:
        print(g1)
        list1 = list(g1.nodes())
        print(f"{k}：{list1}")
        nx.draw_networkx_nodes(g, pos, list1, node_size=20, node_color=colors[k-1], label=str(k))
        k += 1
        g1 = core.k_shell(g, k)
    nx.draw_networkx_edges(g, pos, edge_color='gray', alpha=0.7)
    plt.title('无向网络k-壳分解')
    plt.axis('off')
    plt.legend()
    plt.show()


def pr(g, name, pos=None):
    """pagerank算法"""
    PR = nx.pagerank(g)
    print(PR)
    PR_frame = pd.DataFrame(list(PR.items()), columns=('node', 'pr'))
    PR_frame = PR_frame.sort_values(by='pr', ascending=False)
    PR_frame.to_excel(name + '.xlsx', index=False)
    # top_10 = PR_frame.head(10)
    top_10 = PR_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(PR_frame['node'])
    node_list = list(top_10['node'])
    print("pr", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    draw_center_network(g, node_list, odd, name, pos)


def find_subgraph(g):
    """寻找网络中的子图并统计其个数"""
    subs = []
    # 寻找网络中的所有子图
    for i in range(3, len(g.nodes()) + 1):
        for bunch in list(combinations(g.nodes(), i)):
            h = nx.induced_subgraph(g, bunch)
            if len(h.edges) >= i-1:
                subs.append(h)
    # 统计各结构个数
    res = dict()
    for i in range(len(subs)):
        if subs[i] not in res.keys():
            res[subs[i]] = 1
            for j in subs[i + 1:]:
                if is_isomorphic(subs[i], j):
                    res[subs[i]] += 1
    return res


# def motif_detection(g):
#     """检测网络中的模体"""
#     # 寻找真实网络中的所有子图
#     subs1_counter = find_subgraph(g)
#     # 计算z-core
#     z_cores = []
#     degree_seq = dict(nx.degree(g)).values()
#     for sub in subs1_counter.keys():
#         subs2_counters = [0] * 4
#         # 生成随机网络
#         for i in range(4):
#             ran = nx.configuration_model(degree_seq)
#             ran_counter = find_subgraph(ran)
#             for key in ran_counter.keys():
#                 if is_isomorphic(key, sub):
#                     subs2_counters[i] = ran_counter[key]
#         z = (subs1_counter[sub] - np.mean(subs2_counters)) / np.std(subs2_counters)
#         z_cores.append(z)
#     print(z_cores)
#     ind = z_cores.index(max(z_cores))
#     motif = list(subs1_counter.keys())
#     motif = motif[ind]
#     nx.draw_networkx(motif)
#     plt.show()


