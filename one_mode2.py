import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import pandas as pd


'''def two2one(path1):
    """将二分图转换成单模图"""
    bs = os.listdir(path1)
    bs.sort()
    sims = []
    for b in bs[:71]:
        print(b)
        bi = nx.read_gml(path1 + b)

        top_nodes = [n for n, d in bi.nodes(data=True) if d["bipartite"] == 0]

        for node0 in top_nodes:
            for node1 in top_nodes:
                neighbors0 = [i for i in bi.neighbors(node0)]
                neighbors1 = [i for i in bi.neighbors(node1)]
                sim = len(list(set(neighbors0).intersection(set(neighbors1)))) / len(list(set(neighbors0).union(set(neighbors1))))
                sims.append(sim)

    n, bins, patches = plt.hist(sims)
    plt.show()


def two2one_direct(path1):
    """将二分图转换成单模图"""
    bs = os.listdir(path1)
    bs.sort()
    sims = []
    for b in bs[:60]:
        print(b)
        bi = nx.read_gml(path1 + b)

        top_nodes = [n for n, d in bi.nodes(data=True) if d["bipartite"] == 0]

        for node0 in top_nodes:
            for node1 in top_nodes:
                neighbors0 = [i for i in bi.neighbors(node0)]
                neighbors1 = [i for i in bi.neighbors(node1)]
                sim = len(list(set(neighbors0).intersection(set(neighbors1)))) / len(neighbors1)
                sims.append(sim)

    n, bins, patches = plt.hist(sims)
    plt.show()'''


def two2one(path1, path2, m):
    """将二分图转换成单模图（无向图）"""
    bs = os.listdir(path1)
    bs.sort()
    for b in bs:
        g = nx.Graph()      # 新建一个无向图
        print(b)
        gml_name = re.sub('\\.gml', '', b)
        bi = nx.read_gml(path1 + b)

        bottom_nodes = [n for n, d in bi.nodes(data=True) if d["bipartite"] == 1]

        for node0 in bottom_nodes:
            for node1 in bottom_nodes:
                if node0 != node1:
                    neighbors0 = [i for i in bi.neighbors(node0)]
                    neighbors1 = [i for i in bi.neighbors(node1)]
                    sim = len(list(set(neighbors0).intersection(set(neighbors1)))) / len(list(set(neighbors0).
                                                                                              union(set(neighbors1))))
                    if sim > m:
                        g.add_edge(node0, node1, weight=sim)
        nx.write_gml(g, path2 + f"{m}/{gml_name}.gml")


def two2one_direct(path1, path2, m):
    """将二分图转换成单模图"""
    bs = os.listdir(path1)
    bs.sort()
    for b in bs:
        g = nx.DiGraph()
        print(b)
        gml_name = re.sub('\\.gml', '', b)
        bi = nx.read_gml(path1 + b)

        bottom_nodes = [n for n, d in bi.nodes(data=True) if d["bipartite"] == 1]

        for node0 in bottom_nodes:
            for node1 in bottom_nodes:
                if node0 != node1:
                    neighbors0 = [i for i in bi.neighbors(node0)]
                    neighbors1 = [i for i in bi.neighbors(node1)]
                    sim0 = len(list(set(neighbors0).intersection(set(neighbors1)))) / len(neighbors0)
                    sim1 = len(list(set(neighbors0).intersection(set(neighbors1)))) / len(neighbors1)
                    if sim0 > m:
                        g.add_edge(node0, node1, weight=sim0)
                    if sim1 > m:
                        g.add_edge(node1, node0, weight=sim1)

        nx.write_gml(g, path2 + f"{m}/{gml_name}.gml")


def gml2csv(g, path, file_name, category):
    """将图片输出为节点csv和边csv"""
    node_labels = g.nodes()
    df1 = pd.DataFrame(list(zip(g.nodes(), node_labels)), columns=('Id', 'Label'))
    # print(df1)
    file_name = number_to_rq[int(file_name)]
    df1.to_csv(path+'nodes/'+category+file_name+'.csv', index=False)
    # print(g.edges())
    df2 = pd.DataFrame(g.edges(), columns=('Source', 'Target'))
    df2.to_csv(path+'edges/'+category+file_name+'.csv', index=False)


def gmls2csvs(path1, path2, category):
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        file_name = re.sub('\\.gml', '', g)
        gml2csv(graph, path2, file_name, category)


frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
rq = frame['开始时间'].drop_duplicates()
rq = list(rq)
rq.sort()
number_to_rq = dict(zip(list(range(len(rq))), rq))

folder_path1 = 'G:/jj_st/bipartite/anjdfen_graphs_gml/graph/'
folder_path2 = 'G:/jj_st/one_mode_graph/st/st_gml/graph/'
folder_path3 = 'G:/jj_st/one_mode_graph/st/st_gml/digraph/'
save_path = 'G:/jj_st/one_mode_graph/jj/'
maxs0 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
maxs1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
# for max0 in maxs0:
#     two2one(folder_path1, folder_path2, max0)
for max1 in maxs1:
    two2one_direct(folder_path1, folder_path3, max1)

# for max0 in maxs0[:1]:
#     gmls2csvs(folder_path2+f"{max0}/", save_path, f"graph/{max0}/")
'''for max1 in maxs1:
    gmls2csvs(folder_path3 + f"{max1}/", save_path, f"digraph/{max1}/")'''
