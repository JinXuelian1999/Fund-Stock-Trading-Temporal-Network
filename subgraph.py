import networkx as nx
import os
import pandas as pd


top = 10
frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
rq = frame['开始时间'].drop_duplicates()
rq = list(rq)
rq.sort()


def subgraph(paths, sav_paths):
    for i in range(2):
        result = []   # 记录结果
        gs = os.listdir(paths[i])
        gs.sort()
        for g in gs:
            print(g)
            graph = nx.read_gml(paths[i] + g)
            degrees = nx.degree(graph)
            sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)      # 按度排序
            # print(sorted_degrees)
            if len(sorted_degrees) >= top:
                largest_top = sorted_degrees[:top]
                # print(largest_10)
                largest_nodes = [v for v, d in largest_top]      # 如果节点个数大于等于top，取度最大的top个节点
            else:
                largest_nodes = [v for v, d in sorted_degrees]  # 小于top，全取
            # print(largest_nodes)
            sum_density = 0
            for node in largest_nodes:
                neighbours = nx.neighbors(graph, node)  # 该节点的邻居节点
                sub_nodes = list(neighbours) + [node]
                sub = nx.induced_subgraph(graph, sub_nodes)   # 该节点与其邻居节点构成的子图
                sum_density += nx.density(sub)  # 求子图密度
            if len(largest_nodes) != 0:
                result.append(sum_density / len(largest_nodes))
            else:
                result.append(0)
        print(result)
        df = pd.DataFrame({"date": rq, "subgraph_density": result})
        print(df)
        df.to_csv(sav_paths[i] + 'subgraph_density.csv', index=False)


if __name__ == "__main__":
    folder_paths = [
        'jj/jj_gml/graph/0.1/',
        'st/st_gml/graph/0.1/'
    ]
    save_paths = [
        'jj/',
        'st/'
    ]
    subgraph(folder_paths, save_paths)
