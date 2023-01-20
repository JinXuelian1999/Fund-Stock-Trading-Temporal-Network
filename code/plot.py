import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import undirected_network as un
import numpy as np
import os
import re
from collections import Counter
import matplotlib


def fig1(path1, path2, path3, path4, path5, path6):
    font = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15
    }
    plt.rcParams['font.size'] = 15

    # # 平均度
    # plt.figure(figsize=(12, 8))
    # ax5 = plt.subplot(111)
    # ax5.tick_params(top=True, right=True)  # 设置刻度样式
    # data1 = pd.read_excel(path1 + "股票平均入度.xlsx", sheet_name=0)
    # data2 = pd.read_excel(path1 + "基金平均出度.xlsx", sheet_name=0)
    #
    # x = list(data1["时间"])
    # y1 = list(data1["股票平均入度"])
    # y2 = list(data2["基金平均出度"])
    #
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y2, color="red", label="Fund")
    # plt.plot(x, y1, color="orange", label="Stock")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("Average degree", fontsize=15)
    # plt.title("A", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_A.png', format='png', dpi=500)

    # # 密度
    # plt.figure(figsize=(12, 8))
    # ax6 = plt.subplot(111)
    # ax6.tick_params(top=True, right=True)  # 设置刻度样式
    # data1 = pd.read_excel(path1 + "密度.xlsx", sheet_name=0)
    # data2 = pd.read_excel(path2 + "密度.xlsx", sheet_name=0)
    # data3 = pd.read_excel(path3 + "密度.xlsx", sheet_name=0)
    #
    # x = list(data1["时间"])
    # y1 = list(data1["密度"])
    # y2 = list(data2["密度"])
    # y3 = list(data3["密度"])
    #
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y1, color="green", label="Fund-Stock")
    # plt.plot(x, y2, color="red", label="Fund")
    # plt.plot(x, y3, color="orange", label="Stock")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("Density", fontsize=15)
    # plt.title("B", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_B.png', format='png', dpi=500)
    #
    # # 直径
    # plt.figure(figsize=(12, 8))
    # ax1 = plt.subplot(111)
    # ax1.tick_params(top=True, right=True)   # 设置刻度样式
    # data1 = pd.read_excel(path1 + "直径.xlsx", sheet_name=0)
    # data2 = pd.read_excel(path2 + "直径.xlsx", sheet_name=0)
    # data3 = pd.read_excel(path3 + "直径.xlsx", sheet_name=0)
    # # data4 = pd.read_excel(path1 + "平均最短路径长度.xlsx", sheet_name=0)
    # x = list(data1["时间"])
    # y1 = list(data1["直径"])
    # y2 = list(data2["直径"])
    # y3 = list(data3["直径"])
    # # y4 = list(data4["平均最短路径长度"])
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y1, color="green")
    # plt.plot(x, y2, color="red")
    # plt.plot(x, y3, color="orange")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("Diameter", fontsize=15)
    # plt.title("C", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_C.png', format='png', dpi=500)
    #
    # # 同配系数
    # plt.figure(figsize=(12, 8))
    # ax2 = plt.subplot(111)
    # ax2.tick_params(top=True, right=True)  # 设置刻度样式
    # data1 = pd.read_excel(path1 + "同配系数.xlsx", sheet_name=0)
    # data2 = pd.read_excel(path2 + "同配系数.xlsx", sheet_name=0)
    # data3 = pd.read_excel(path3 + "同配系数.xlsx", sheet_name=0)
    # x = list(data1["时间"])
    # y1 = list(data1["同配系数"])
    # y2 = list(data2["同配系数"])
    # y3 = list(data3["同配系数"])
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y1, color="green")
    # plt.plot(x, y2, color="red")
    # plt.plot(x, y3, color="orange")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("r", fontsize=15)
    # plt.title("D", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_D.png', format='png', dpi=500)
    #
    # # 平均聚类系数
    # plt.figure(figsize=(12, 8))
    # ax3 = plt.subplot(111)
    # ax3.tick_params(top=True, right=True)  # 设置刻度样式
    # data1 = pd.read_excel(path1 + "平均聚类系数.xlsx", sheet_name=0)
    # data2 = pd.read_excel(path2 + "平均聚类系数.xlsx", sheet_name=0)
    # data3 = pd.read_excel(path3 + "平均聚类系数.xlsx", sheet_name=0)
    # x = list(data1["时间"])
    # y1 = list(data1["平均聚类系数"])
    # y2 = list(data2["平均聚类系数"])
    # y3 = list(data3["平均聚类系数"])
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y1, color="green")
    # plt.plot(x, y2, color="red")
    # plt.plot(x, y3, color="orange")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("Average Clustering Coefficient", fontsize=15)
    # plt.title("E", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_E.png', format='png', dpi=500)
    #
    # 度分布
    plt.figure(figsize=(12, 8))
    ax4 = plt.subplot(111)
    ax4.tick_params(top=True, right=True)  # 设置刻度样式
    # ax4.spines['right'].set_visible(False)
    # ax4.spines['top'].set_visible(False)
    # ax4.tick_params(direction="in")
    g = nx.read_gml(path4)
    in_counts = un.in_degree_histogram(g)
    out_counts = un.out_degree_histogram(g)

    in_p = [i / sum(in_counts) for i in in_counts]
    out_p = [i / sum(out_counts) for i in out_counts]

    x1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(in_counts))]
    y1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in in_p]

    x2 = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(out_counts))]
    y2 = [pow(10, np.log10(i)) if i > 0 else 0 for i in out_p]

    plt.scatter(x1, y1, c="green", marker="o", alpha=0.3, linewidths=1, label="in-degree")
    plt.scatter(x2, y2, c="red", marker="s", alpha=0.3, linewidths=1, label="out-degree")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("k", fontsize=15)
    plt.ylabel("P(k)", fontsize=15)
    bm = re.sub('\\.gml', '', path4.split('/')[-1])
    plt.title(rq[int(bm)], fontsize="x-large", fontweight="black")
    # plt.title("H", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.tight_layout(pad=12, w_pad=5, h_pad=12)
    plt.savefig(f'G:/jj_st/写论文/度分布/{rq[int(bm)]}.png', format='png', dpi=500)

    # 网络节点重要性
    # 设置画布的大小
    # figure_names = ['K', 'L', 'M']
    # for i in range(3):
    #     top_one = []
    #     files = os.listdir(path5[i])
    #     files.sort()
    #     for file in files:
    #         print(file)
    #         data = pd.read_csv(path5[i] + file, low_memory=False)
    #         data.dropna(how='any', inplace=True)
    #         data.index = range(data.shape[0])
    #         try:
    #             top_one.append(data.loc[0, "type or industry"])
    #         except KeyError:
    #             continue
    #     # print(top_one)
    #     print(len(top_one))
    #     counter = Counter(top_one)
    #     print(counter)
    #     counter_order = dict(sorted(counter.items(), key=lambda x: x[1], reverse=False))
    #     print(counter_order)
    #     print(counter_order.values())
    #     colors = matplotlib.cm.GnBu(np.arange(len(counter_order)) / len(counter_order))
    #     explode = [0] * (len(counter_order)-1) + [0.1]
    #     fig = plt.figure(figsize=(22, 10))
    #     ax = fig.add_axes([0.1, 0.05, 0.5, 0.99])
    #     # 第一个传入的是我们需要计算的数据，
    #     ax.pie(counter_order.values(), colors=colors, autopct=autopct(), explode=explode)
    #     # 绘图的标题和图例
    #     plt.title(figure_names[i], fontsize="xx-large", fontweight="black", x=-0.08, y=0.85)
    #     plt.legend(labels=counter_order.keys(), frameon=False, prop=font, labelspacing=0.1, bbox_to_anchor=(1.05, 0.5),
    #                loc=6, borderaxespad=0)
    #     # 存储图片
    #     # plt.show()
    #     plt.savefig("G:/jj_st/写论文/" + f"figure1_网络节点重要性{figure_names[i]}.png", format='png', dpi=500)

    # # 箱线图-度分布
    # node_in_degree = []
    # node_out_degree = []
    # gs = os.listdir(path6)
    # gs.sort()
    # for g in gs:
    #     print(g)
    #     graph = nx.read_gml(path6 + g)
    #     node_in_degree.append([d for n, d in graph.in_degree()])
    #     node_out_degree.append([d for n, d in graph.out_degree()])
    #
    # plt.figure(figsize=(62, 22))
    # ax7 = plt.subplot(111)
    # ax7.tick_params(top=True, right=True)  # 设置刻度样式
    # bplot1 = ax7.boxplot(node_in_degree, vert=True, patch_artist=True, notch=True, showfliers=False)
    # for patch in bplot1['boxes']:
    #     patch.set_facecolor('lightblue')
    # plt.xticks(range(80), rq, rotation='vertical', fontsize=15)
    # plt.xlabel("Time(quarter)", fontsize=50)
    # plt.ylabel("In-Degree", fontsize=50)
    # plt.title("I", fontsize=100, fontweight="black", x=-0.08, y=0.95)
    # plt.savefig('G:/jj_st/写论文/figure1_I.png', format='png', dpi=500)
    #
    # plt.figure(figsize=(62, 22))
    # ax8 = plt.subplot(111)
    # ax8.tick_params(top=True, right=True)  # 设置刻度样式
    # bplot2 = ax8.boxplot(node_out_degree, vert=True, patch_artist=True, notch=True, showfliers=False)
    # for patch in bplot2['boxes']:
    #     patch.set_facecolor('lightgreen')
    # plt.xticks(range(80), rq, rotation='vertical', fontsize=15)
    # plt.xlabel("Time(quarter)", fontsize=50)
    # plt.ylabel("Out-Degree", fontsize=50)
    # plt.title("J", fontsize=100, fontweight="black", x=-0.08, y=0.95)
    # # plt.tight_layout()
    # # plt.show()
    # plt.savefig('G:/jj_st/写论文/figure1_J.png', format='png', dpi=500)

    # # 包含度最大节点的子图的平均密度（基金）
    # plt.figure(figsize=(12, 8))
    # ax9 = plt.subplot(111)
    # ax9.tick_params(top=True, right=True)  # 设置刻度样式
    # data1 = pd.read_csv(path2 + "subgraph_density_top=10.csv")
    # data2 = pd.read_csv(path2 + "subgraph_density_top=5.csv")
    # data3 = pd.read_csv(path2 + "subgraph_density_top=1.csv")
    #
    # x = list(data1["date"])
    # y1 = list(data1["subgraph_density"])
    # y2 = list(data2["subgraph_density"])
    # y3 = list(data3["subgraph_density"])
    #
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y1, color="green", label="Top 10 nodes")
    # plt.plot(x, y2, color="red", label="Top 5 nodes")
    # plt.plot(x, y3, color="orange", label="Top 1 node")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("Mean density of fund subgraph(s)", fontsize=15)
    # plt.title("F", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_F.png', format='png', dpi=500)
    #
    # # 包含度最大节点的子图的平均密度（股票）
    # plt.figure(figsize=(12, 8))
    # ax10 = plt.subplot(111)
    # ax10.tick_params(top=True, right=True)  # 设置刻度样式
    # data1 = pd.read_csv(path3 + "subgraph_density_top=10.csv")
    # data2 = pd.read_csv(path3 + "subgraph_density_top=5.csv")
    # data3 = pd.read_csv(path3 + "subgraph_density_top=1.csv")
    #
    # x = list(data1["date"])
    # y1 = list(data1["subgraph_density"])
    # y2 = list(data2["subgraph_density"])
    # y3 = list(data3["subgraph_density"])
    #
    # plt.xticks(range(80), x, rotation='vertical', fontsize=6)
    # plt.plot(x, y1, color="green")
    # plt.plot(x, y2, color="red")
    # plt.plot(x, y3, color="orange")
    # plt.xlabel("Time(quarter)", fontsize=15)
    # plt.ylabel("Mean density of stock subgraph(s)", fontsize=15)
    # plt.title("G", fontsize="xx-large", fontweight="black", x=-0.08, y=0.95)
    # # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.savefig('G:/jj_st/写论文/figure1_G.png', format='png', dpi=500)


def autopct():
    def my_autopct(pct):
        if pct > 5:
            return '{p:.2f}%'.format(p=pct)
        else:
            return ''
    return my_autopct


def fig2(path1, path2):
    font = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 25
    }
    plt.rcParams['font.size'] = 25

    # # fig1
    # plt.figure(figsize=(12, 8))
    # ax1 = plt.subplot(111)
    # ax1.tick_params(top=True, right=True)
    # # ax4.spines['right'].set_visible(False)
    # # ax4.spines['top'].set_visible(False)
    # # ax4.tick_params(direction="in")
    # g = nx.read_gml(path1)
    # in_counts = un.in_degree_histogram(g)
    #
    # in_p = [i / sum(in_counts) for i in in_counts]
    #
    # x1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(in_counts))]
    # y1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in in_p]
    #
    # plt.scatter(x1, y1, c="green", marker="o", linewidths=0.8)
    # plt.plot(range(1, 50), [11/36*x**(-29/18) for x in range(1, 50)], color='black', linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("k", fontsize=25)
    # plt.ylabel("P(k)", fontsize=25)
    # plt.title("A", fontsize="xx-large", fontweight="black", x=-0.12, y=0.95)
    # # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # # plt.tight_layout(pad=12, w_pad=5, h_pad=12)
    # plt.savefig('G:/jj_st/写论文/figure2_A.png', format='png', dpi=500)

    # fig2
    plt.figure(figsize=(12, 8))
    ax2 = plt.subplot(111)
    ax2.tick_params(top=True, right=True)
    # ax4.spines['right'].set_visible(False)
    # ax4.spines['top'].set_visible(False)
    # ax4.tick_params(direction="in")
    g = nx.read_gml(path2)
    in_counts = un.in_degree_histogram(g)

    in_p = [i / sum(in_counts) for i in in_counts]

    x1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in range(len(in_counts))]
    y1 = [pow(10, np.log10(i)) if i > 0 else 0 for i in in_p]

    m = 3
    n = 3
    l = 2
    coef = (m+n)*n**((m+n)/(m+4*l))/(2*(m+4*l))
    expo = (-1)*(m+n)/(m+4*l)-1
    print(coef, expo)
    plt.scatter(x1, y1, c="green", marker="o", linewidths=0.8)
    plt.plot(range(2, 70), [coef * x ** expo for x in range(2, 70)], color='black', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("k", fontsize=25)
    plt.ylabel("P(k)", fontsize=25)
    plt.title("B", fontsize="xx-large", fontweight="black", x=-0.12, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # plt.tight_layout(pad=12, w_pad=5, h_pad=12)
    plt.savefig('G:/jj_st/写论文/figure2_B.png', format='png', dpi=500)

    # # fig3
    # plt.figure(figsize=(12, 8))
    # ax3 = plt.subplot(111)
    # ax3.tick_params(top=True, right=True)
    # # ax4.spines['right'].set_visible(False)
    # # ax4.spines['top'].set_visible(False)
    # # ax4.tick_params(direction="in")
    # linestyles = ['--', '-', '-.']
    # n = 1
    # l = 2
    # ms = [5, 10, 15]
    # cs = []
    # gammas = []
    # for m in ms:
    #     coef = (m+n)*n**((m+n)/(m+4*l))/(2*(m+4*l))
    #     expo = (-1)*(m+n)/(m+4*l)-1
    #     cs.append(coef)
    #     gammas.append(expo)
    # for i in range(3):
    #     plt.plot(range(1, 1000), [cs[i] * x ** gammas[i] for x in range(1, 1000)], color='black', linestyle=linestyles[i],
    #              label=f"m={ms[i]}")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("k", fontsize=25)
    # plt.ylabel("P(k)", fontsize=25)
    # plt.title("C", fontsize="xx-large", fontweight="black", x=-0.12, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # # plt.tight_layout(pad=12, w_pad=5, h_pad=12)
    # plt.savefig('G:/jj_st/写论文/figure2_C.png', format='png', dpi=500)
    #
    # # fig4
    # plt.figure(figsize=(12, 8))
    # ax4 = plt.subplot(111)
    # ax4.tick_params(top=True, right=True)
    # # ax4.spines['right'].set_visible(False)
    # # ax4.spines['top'].set_visible(False)
    # # ax4.tick_params(direction="in")
    # linestyles = ['--', '-', '-.']
    # ns = [1, 5, 10]
    # l = 2
    # m = 10
    # cs = []
    # gammas = []
    # for n in ns:
    #     coef = (m + n) * n ** ((m + n) / (m + 4 * l)) / (2 * (m + 4 * l))
    #     expo = (-1) * (m + n) / (m + 4 * l) - 1
    #     cs.append(coef)
    #     gammas.append(expo)
    # for i in range(3):
    #     plt.plot(range(1, 1000), [cs[i] * x ** gammas[i] for x in range(1, 1000)], color='black',
    #              linestyle=linestyles[i],
    #              label=f"n={ns[i]}")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("k", fontsize=25)
    # plt.ylabel("P(k)", fontsize=25)
    # plt.title("D", fontsize="xx-large", fontweight="black", x=-0.12, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # # plt.tight_layout(pad=12, w_pad=5, h_pad=12)
    # plt.savefig('G:/jj_st/写论文/figure2_D.png', format='png', dpi=500)
    #
    # # fig5
    # plt.figure(figsize=(12, 8))
    # ax5 = plt.subplot(111)
    # ax5.tick_params(top=True, right=True)
    # # ax4.spines['right'].set_visible(False)
    # # ax4.spines['top'].set_visible(False)
    # # ax4.tick_params(direction="in")
    # linestyles = ['--', '-', '-.']
    # ls = [1, 2, 3]
    # n = 1
    # m = 10
    # cs = []
    # gammas = []
    # for l in ls:
    #     coef = (m + n) * n ** ((m + n) / (m + 4 * l)) / (2 * (m + 4 * l))
    #     expo = (-1) * (m + n) / (m + 4 * l) - 1
    #     cs.append(coef)
    #     gammas.append(expo)
    # for i in range(3):
    #     plt.plot(range(1, 1000), [cs[i] * x ** gammas[i] for x in range(1, 1000)], color='black',
    #              linestyle=linestyles[i],
    #              label=f"l={ls[i]}")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("k", fontsize=25)
    # plt.ylabel("P(k)", fontsize=25)
    # plt.title("E", fontsize="xx-large", fontweight="black", x=-0.12, y=0.95)
    # plt.legend(frameon=False, prop=font, labelspacing=0.1)
    # # plt.tight_layout(pad=12, w_pad=5, h_pad=12)
    # plt.savefig('G:/jj_st/写论文/figure2_E.png', format='png', dpi=500)


if __name__ == "__main__":
    frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
    rq = frame['开始时间'].drop_duplicates()
    rq = list(rq)
    rq.sort()
    folder_path1 = 'G:/jj_st/bipartite/'
    folder_path2 = 'G:/jj_st/one_mode_graph/jj/'
    folder_path3 = 'G:/jj_st/one_mode_graph/st/'
    folder_path4 = 'G:/jj_st/bipartite/anjdfen_graphs_gml/digraph/79.gml'
    folder_path5 = 'G:/jj_st/写论文/T=5000，N_F0=2，N_S0=15，m=10，n=1，l=2.gml'
    folder_path6 = ["G:/jj_st/bipartite/duzxx_table/", "G:/jj_st/bipartite/jjzxx_table/",
                    "G:/jj_st/bipartite/jszxx_table/"]
    folder_path7 = 'G:/jj_st/bipartite/anjdfen_graphs_gml/digraph/'
    # fig1(folder_path1, folder_path2, folder_path3, folder_path4, folder_path6, folder_path7)
    # fig2(folder_path5, folder_path4)
    path = 'G:/jj_st/bipartite/anjdfen_graphs_gml/digraph/'
    files = os.listdir(path)
    files.sort()
    for file in files:
        print(file)
        fig1(folder_path1, folder_path2, folder_path3, path + file, folder_path6, folder_path7)
