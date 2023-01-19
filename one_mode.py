import os
from networkx.algorithms.bipartite.projection import projected_graph
import networkx as nx
import matplotlib.pyplot as plt
import re
import undirected_network as un
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from networkx.algorithms import centrality
from networkx.algorithms import community
from pyecharts import options as opts
from pyecharts.charts import Graph


def two2one(path1, path2, mode=1):
    """将二分图转换成单模图"""
    bs = os.listdir(path1)
    bs.sort()
    for b in bs:
        print(b)
        bi = nx.read_gml(path1 + b)
        if mode == 1:
            top_nodes = {n for n, d in bi.nodes(data=True) if d["bipartite"] == 0}
            one_mode = projected_graph(bi, top_nodes)
        else:
            bottom_nodes = {n for n, d in bi.nodes(data=True) if d["bipartite"] == 1}
            one_mode = projected_graph(bi, bottom_nodes)
        # print(one_mode.edges())
        nx.write_gml(one_mode, path2 + f"{b}")


def diameters(path1, pic_name, path2):
    """计算80个季度的直径并绘制折线图"""
    d_list = []
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        d = un.diameter(graph)
        # print(d)
        d_list.append(d)
    dtf = pd.DataFrame({'时间': rq, '直径': d_list})
    dtf.to_excel(path2+"直径.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(d_list) + 1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, d_list)
    plt.savefig(path2+pic_name+'.pdf', format='pdf', dpi=500)


def ave_degree(path1, pic_name, path2):
    """计算80个季度的平均度并绘制折线图"""
    ave_degree_list = []
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        # print(g)
        graph = nx.read_gml(path1 + g)
        a_d = un.average_degree(graph)
        # print(a_d)
        ave_degree_list.append(a_d)
    # a_s = pd.Series(ave_degree_list, index=rq)
    dtf = pd.DataFrame({'时间': rq, '平均度': ave_degree_list})
    dtf.to_excel(path2+"平均度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_degree_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, ave_degree_list)
    # a_s.plot(rot='60')
    plt.savefig(path2+pic_name+'.pdf', format='pdf', dpi=500)


def degree_d(path1, path2, co_to_date, pic_name):
    """计算80个季度的度分布并绘制折线图"""
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        un.degree_distribution(graph, path2, re.sub('\\.gml', '', g), co_to_date, pic_name)


def assortativity(path1, pic_name, path2):
    """计算80个季度的同配性并绘制折线图"""
    r_list = []
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        r = nx.degree_assortativity_coefficient(graph)
        r_list.append(r)
    dtf = pd.DataFrame({'时间': rq, '同配系数': r_list})
    dtf.to_excel(path2+"同配系数.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(r_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, r_list)
    # a_s.plot(rot='60')
    plt.savefig(path2+pic_name+'.pdf', format='pdf', dpi=500)


def densities(path1, pic_name, path2):
    """计算80个季度的密度并绘制折线图"""
    den_list = []
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        den = nx.density(graph)
        # print(a_d)
        den_list.append(den)
    dtf = pd.DataFrame({'时间': rq, '密度': den_list})
    dtf.to_excel(path2+"密度.xlsx", index=False)
    plt.figure(figsize=(22, 12))
    x = list(range(1, len(den_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, den_list)
    # a_s.plot(rot='60')
    plt.savefig(path2+pic_name+'.pdf', format='pdf', dpi=500)


def clustering(path1, pic_name, path2):
    """计算20个季度的平均聚类系数并绘制折线图"""
    ave_cluster_list = []
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        try:
            ave_cluster = nx.average_clustering(graph)
            # print(a_d)
        except ZeroDivisionError:
            ave_cluster = -1
        ave_cluster_list.append(ave_cluster)
    dtf = pd.DataFrame({'时间': rq, '平均聚类系数': ave_cluster_list})
    dtf.to_excel(path2+"平均聚类系数.xlsx", index=False)

    plt.figure(figsize=(22, 12))
    x = list(range(1, len(ave_cluster_list)+1))
    plt.xticks(x, rq, rotation='vertical')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(pic_name, fontsize='xx-large')
    plt.plot(x, ave_cluster_list)
    # a_s.plot(rot='60')
    plt.savefig(path2+pic_name+'.pdf', format='pdf', dpi=500)


def convert(l):
    t = []
    for item in l:
        t.append((int(item[0]), int(item[1])))
    return t


def draw_net(path1, filename, pos, path2):
    """画图"""
    g = nx.read_gml(path1 + filename)
    plt.figure(figsize=(50, 30))
    # print(pos)
    nx.draw_networkx_nodes(g, pos, list(map(int, g.nodes())), node_size=1)
    nx.draw_networkx_edges(g, pos, convert(g.edges()), width=0.01)
    # print(convert(g.edges()))
    plt.axis('off')
    pic_name = re.sub('\\.gml', '', filename)
    title = number_to_rq[int(pic_name)]
    plt.title(title, fontsize=60)
    plt.savefig(path2 + f"{pic_name}.png", format='png', dpi=70)


def draw_nets(path1, path2, pos):
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        draw_net(path1, g, pos, path2)


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
    nx.draw_networkx_edges(g, pos, convert(g.edges()), width=0.01)
    plt.axis('off')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    title = co_to_date[int(net_name)]
    plt.title(title, fontsize=60)
    # plt.show()
    plt.savefig(savefig_path + f"{net_name}.png", format='png', dpi=70)


def degree_center(g, name, co_to_date, path, delta, pos=None):
    """计算节点的度中心性"""
    node_degree_centrality = centrality.degree_centrality(g)
    degree_center_frame = pd.DataFrame(list(node_degree_centrality.items()), columns=('node', 'degree_centrality'))
    degree_center_frame = degree_center_frame.sort_values(by='degree_centrality', ascending=False)
    # print(degree_center_frame)
    if not os.path.exists(path + f"{delta}/"):
        os.mkdir(path + f"{delta}/")
    degree_center_frame.to_csv(path + f"{delta}/" + co_to_date[int(name)] + '.csv', index=False)
    '''top_10 = degree_center_frame.head(10)
    # top_10 = degree_center_frame.head(round(g.number_of_nodes()*0.05))
    all_node_list = list(degree_center_frame['node'])
    node_list = list(top_10['node'])
    # print("度中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    fig_path = path + "duzxx/"
    draw_center_network(g, node_list, odd, name, fig_path, co_to_date, pos)'''


def betweenness_center(g, name, co_to_date, path, delta, pos=None):
    """计算节点的介数中心性"""
    node_betweenness_centrality = centrality.betweenness_centrality(g)
    # print(node_betweenness_centrality)
    betweenness_centrality_frame = pd.DataFrame(list(node_betweenness_centrality.items()), columns=(
        'node', 'betweenness_centrality'))
    # print(node_betweenness_centrality_frame)
    betweenness_centrality_frame = betweenness_centrality_frame.sort_values(by='betweenness_centrality',
                                                                            ascending=False)
    if not os.path.exists(path + f"{delta}/"):
        os.mkdir(path + f"{delta}/")
    betweenness_centrality_frame.to_csv(path + f"{delta}/" + co_to_date[int(name)] + '.csv', index=False)
    '''top_10 = betweenness_centrality_frame.head(10)
    # top_10 = node_betweenness_centrality_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(betweenness_centrality_frame['node'])
    node_list = list(top_10['node'])
    # print("介数中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    fig_path = path + 'jszxx/'
    draw_center_network(g, node_list, odd, name, fig_path, co_to_date, pos)'''


def close_to_the_center(g, name, co_to_date, path, delta, pos=None):
    """计算节点的接近中心性"""
    node_closeness_centrality = centrality.closeness_centrality(g)
    closeness_centrality_frame = pd.DataFrame(list(node_closeness_centrality.items()), columns=(
        'node', 'closeness_centrality'))
    closeness_centrality_frame = closeness_centrality_frame.sort_values(by='closeness_centrality', ascending=False)
    if not os.path.exists(path + f"{delta}/"):
        os.mkdir(path + f"{delta}/")
    closeness_centrality_frame.to_csv(path + f"{delta}/" + co_to_date[int(name)] + '.csv', index=False)
    '''top_10 = closeness_centrality_frame.head(10)
    # top_10 = close_to_the_center_frame.head(round(g.number_of_nodes() * 0.05))
    all_node_list = list(closeness_centrality_frame['node'])
    node_list = list(top_10['node'])
    # print("接近中心性：", node_list)
    odd = diff_of_two_list(all_node_list, node_list)
    fig_path = path + 'jjzxx/'
    draw_center_network(g, node_list, odd, name, fig_path, co_to_date, pos)'''


def center(path1, path2, delta, pos=None):
    """计算每张图的中心性"""
    gs = os.listdir(path1)
    gs.sort()
    for g in gs[78:]:
        print(g)
        graph = nx.read_gml(path1 + g)
        pic_name = re.sub('\\.gml', '', g)
        degree_center(graph, pic_name, number_to_rq, path2+'duzxx_table/graph/', delta, pos)
        betweenness_center(graph, pic_name, number_to_rq, path2+'jszxx_table/graph/', delta, pos)
        close_to_the_center(graph, pic_name, number_to_rq, path2+'jjzxx_table/graph/', delta, pos)


def cnm(path1, path2, co_to_date, pos=None):
    """利用CNM算法为每张图划分社团"""
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        if len(graph.nodes()) > 0:
            pic_name = re.sub('\\.gml', '', g)
            communities = community.greedy_modularity_communities(graph)
            # print(communities)
            communities = [list(com) for com in communities]
            # print(communities)
            communities_frame = pd.DataFrame({'members': communities})
            communities_frame.index = [i+1 for i in communities_frame.index]
            communities_frame.index.name = 'community'
            communities_frame.to_csv(path2 + co_to_date[int(pic_name)] + '.csv')
            '''plt.figure(figsize=(50, 30))
            if pos is None:
                pos = nx.spring_layout(g)
            cnt = 0
            for com in communities:
                cnt += 1
                list_nodes = list(com)
                nx.draw_networkx_nodes(graph, pos, list(map(int, list_nodes)), node_size=1, node_color=un.random_color(),
                                       label=f"community{cnt}")
            nx.draw_networkx_edges(graph, pos, convert(graph.edges()), width=0.01)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            title = co_to_date[int(pic_name)]
            plt.title(title, fontsize=60)
            plt.legend(fontsize=35, markerscale=15)
            plt.axis('off')
    
            plt.savefig(path2 + 'communities/' + f"{pic_name}.png", format='png', dpi=70)'''


def force_graph(g, pic_name):
    """绘制力传导图"""
    graph = (
        Graph(init_opts=opts.InitOpts(width='1000px', height='800px')).add("", g.nodes(), g.edges(),
                                                                           layout="force").set_global_opts(
            title_opts=opts.TitleOpts(title=number_to_rq[int(pic_name)]),
            toolbox_opts=opts.ToolBoxFeatureSaveAsImageOpts(type_="png", name=pic_name)
        )
    )
    graph.render("力传导网络图.html")


def force_graphs(path):
    """计算每张图的中心性"""
    gs = os.listdir(path)
    gs.sort()
    for g in gs[10:11]:
        print(g)
        graph = nx.read_gml(path + g)
        pic_name = re.sub('\\.gml', '', g)
        force_graph(graph, pic_name)


def gml2csv(g, path, file_name):
    """将图片输出为节点csv和边csv"""
    node_labels = g.nodes()
    df1 = pd.DataFrame(list(zip(g.nodes(), node_labels)), columns=('Id', 'Label'))
    # print(df1)
    df1.to_csv(path+'nodes/'+file_name+'.csv', index=False)
    # print(g.edges())
    df2 = pd.DataFrame(g.edges(), columns=('Source', 'Target'))
    df2.to_csv(path+'edges/'+file_name+'.csv', index=False)


def gmls2csvs(path1, path2):
    gs = os.listdir(path1)
    gs.sort()
    for g in gs:
        print(g)
        graph = nx.read_gml(path1 + g)
        file_name = re.sub('\\.gml', '', g)
        gml2csv(graph, path2, file_name)


def jj_id_char(file1, file2, file3):
    """输出基金名称与类型"""
    data1 = pd.read_csv(file1, low_memory=False)
    data2 = pd.read_csv(file2, usecols=['jjid', '基金名称'], low_memory=False)
    data3 = pd.read_csv(file3, usecols=['基金名称', '基金类型'], low_memory=False)

    data1['members'] = data1['members'].map(eval)
    # print(data1['members'].loc[0])
    communities = list(data1['members'])
    data1 = data1.set_index('community')
    n = 1
    for com in communities:
        jj_names = []
        jj_type = []
        for mem in com:
            print(mem)
            frame1 = data2.where(data2['jjid'] == int(mem)).dropna(how='all').drop_duplicates()
            frame1 = frame1.set_index('jjid')
            print(frame1['基金名称'].loc[int(mem)])
            jj_names.append(frame1['基金名称'].loc[int(mem)])
            # print(jj_names)
            frame2 = data3.where(data3['基金名称'] == jj_names[-1]).dropna(how='all').drop_duplicates()
            frame2 = frame2.set_index('基金名称')
            print(frame2['基金类型'].loc[jj_names[-1]])
            jj_type.append(frame2['基金类型'].loc[jj_names[-1]])
        data1.loc[n, '基金名称'] = str(jj_names)
        data1.loc[n, '基金类型'] = str(jj_type)
        n += 1
    data1.to_csv(file1)


def jj_id2char(path, filename1, filename2):
    files = os.listdir(path)
    files.sort()
    for file in files:
        print(file)
        jj_id_char(path+file, filename1, filename2)


def st_id_char(file1, file2, file3):
    """输出股票名称与行业"""
    data1 = pd.read_csv(file1, low_memory=False)
    data2 = pd.read_csv(file2, usecols=['stid', '股票名称', '股票代码'], low_memory=False)
    data3 = pd.read_csv(file3, usecols=['Stkcd', 'Nindnme'], low_memory=False)

    data1['members'] = data1['members'].map(eval)
    # print(data1['members'].loc[0])
    communities = list(data1['members'])
    data1 = data1.set_index('community')
    n = 1
    for com in communities:
        st_names = []
        st_codes = []
        st_ind = []
        for mem in com:
            print(mem)
            frame1 = data2.where(data2['stid'] == int(mem)).dropna(how='all').drop_duplicates().head(1)
            frame1 = frame1.set_index('stid')
            print(frame1['股票名称'].loc[int(mem)])
            st_names.append(frame1['股票名称'].loc[int(mem)])
            try:
                print(frame1['股票代码'].loc[int(mem)])
                st_codes.append(int(frame1['股票代码'].loc[int(mem)]))
                frame2 = data3.where(data3['Stkcd'] == st_codes[-1]).dropna(how='all').drop_duplicates()
                frame2 = frame2.set_index('Stkcd')
                print(frame2['Nindnme'].loc[st_codes[-1]])
                st_ind.append(frame2['Nindnme'].loc[st_codes[-1]])
            except ValueError:
                if len(st_names) != 0:
                    st_names.pop()
                continue
            except KeyError:
                if len(st_names) != 0 and len(st_codes) != 0:
                    st_names.pop()
                    st_codes.pop()
                continue
        data1.loc[n, '股票名称'] = str(st_names)
        data1.loc[n, '股票行业'] = str(st_ind)
        n += 1
    data1.to_csv(file1)


def st_id2char(path, filename1, filename2):
    files = os.listdir(path)
    files.sort()
    # with ThreadPoolExecutor(50) as t:
    for file in files:
        print(file)
        st_id_char(path+file, filename1, filename2)


def jj_center_name_type(path, filename1, filename2):
    """基金码-中心性-基金名称-基金类型"""
    # 读取文件
    data1 = pd.read_csv(filename1, low_memory=False)
    data2 = pd.read_csv(filename2, low_memory=False)
    # 构造字典
    jj_ids = list(data1['jjid'])
    jj_names1 = list(data1['基金名称'])
    id_to_name = dict(zip(jj_ids, jj_names1))
    jj_names2 = list(data2['基金名称'])
    jj_types = list(data2['基金类型'])
    name_to_type = dict(zip(jj_names2, jj_types))

    # 中心性文件
    files = os.listdir(path)
    files.sort()
    for file in files:
        print(file)
        data3 = pd.read_csv(path+file, low_memory=False)
        data3['name'] = data3['node'].map(id_to_name)
        data3['type'] = data3['name'].map(name_to_type)
        data3.to_csv(f"{path}{file}", index=False)


def st_center_name_ind(path, filename1, filename2):
    """股票码-中心性-股票名称-股票行业"""
    # 读取文件
    data1 = pd.read_csv(filename1, low_memory=False)
    data2 = pd.read_csv(filename2, low_memory=False)
    # 构造字典
    st_ids = list(data1['stid'])
    st_names = list(data1['股票名称'])
    id_to_name = dict(zip(st_ids, st_names))
    st_codes1 = list(data1['股票代码'].fillna(value=0).astype('int'))
    id_to_code = dict(zip(st_ids, st_codes1))
    st_code2 = list(data2['Stkcd'])
    st_ind = list(data2['Nindnme'])
    code_to_ind = dict(zip(st_code2, st_ind))

    # 中心性文件
    files = os.listdir(path)
    files.sort()
    for file in files:
        print(file)
        data3 = pd.read_csv(path+file, low_memory=False)
        data3['name'] = data3['node'].map(id_to_name)
        data3['code'] = data3['node'].map(id_to_code)
        data3['industry'] = data3['code'].map(code_to_ind)
        data3.to_csv(f"{path}{file}", index=False)


folder_path1 = 'G:/jj_st/bipartite/anjdfen_graphs_gml/'
folder_path2 = 'G:/jj_st/one_mode_graph/jj/jj_gml/graph/0.1/'
folder_path3 = 'G:/jj_st/one_mode_graph/st/st_gml/graph/0.1/'
save_path1 = 'G:/jj_st/one_mode_graph/jj/'
save_path2 = 'G:/jj_st/one_mode_graph/st/'
degree_path1 = 'G:/jj_st/one_mode_graph/jj/degree_distribution/单log/'
degree_path2 = 'G:/jj_st/one_mode_graph/st/degree_distribution/单log/'

frame = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
rq = frame['开始时间'].drop_duplicates()
rq = list(rq)
rq.sort()
number_to_rq = dict(zip(list(range(len(rq))), rq))

# 二分图-->单模图
# two2one(folder_path1, folder_path2, mode=1)
# two2one(folder_path1, folder_path3, mode=2)
# 直径
# name = '基金网络直径变化情况'
# diameters(folder_path2, name, save_path1)
# name = '股票网络直径变化情况'
# diameters(folder_path3, name, save_path2)
# 平均度
# name = '基金网络平均度变化情况'
# ave_degree(folder_path2, name, save_path1)
# name = '股票网络平均度变化情况'
# ave_degree(folder_path3, name, save_path2)
# 度分布
# name = '基金网络度分布'
# degree_d(folder_path2, degree_path1, number_to_rq, name)
# name = '股票网络度分布'
# degree_d(folder_path3, degree_path2, number_to_rq, name)
# 同配系数
# name = '基金网络同配系数变化情况'
# assortativity(folder_path2, name, save_path1)
# name = '股票网络同配系数变化情况'
# assortativity(folder_path3, name, save_path2)
# 密度
# name = '基金网络密度变化情况'
# densities(folder_path2, name, save_path1)
# name = '股票网络密度变化情况'
# densities(folder_path3, name, save_path2)
# 平均聚类系数
# name = '基金网络平均聚类系数变化情况'
# clustering(folder_path2, name, save_path1)
# name = '股票网络平均聚类系数变化情况'
# clustering(folder_path3, name, save_path2)

'''df = pd.read_csv('jjst.csv', usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
jj = list(df['jjid'].drop_duplicates())
st = list(df['stid'].drop_duplicates())
n = len(jj) + len(st)
x = np.divide(list(np.random.randint(1, 200, size=n)), 4)
y = np.divide(list(np.random.randint(1, 120, size=n)), 4)
position = list(zip(x, y))
nodes_position_frame = pd.DataFrame({'node_position': position})
nodes_position_frame.index.name = 'node'
nodes_position_frame.to_csv('G:/jj_st/one_mode_graph/nodes_position.csv')
position_frame = pd.read_csv('G:/jj_st/one_mode_graph/nodes_position.csv')
position = dict(zip(position_frame['node'], position_frame['node_position'].map(eval)))'''
# print(position)
# 网络
# draw_nets(folder_path2, save_path1 + 'nets/', position)
# draw_nets(folder_path3, save_path2 + 'nets/', position)
# 中心性
# center(folder_path2, save_path1, position)
# center(folder_path3, save_path2, position)
# 社团
# cnm(folder_path2, save_path1, number_to_rq, position)
# cnm(folder_path3, save_path2, number_to_rq, position)

# gml转为csv文件
# gmls2csvs(folder_path2, save_path1)
# gmls2csvs(folder_path3, save_path2)


# 社团名单
maxs0 = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
maxs1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9]

# 基金
# for max0 in maxs0[:1]:
#     center(folder_path2+f"graph/{max0}/", save_path1, max0)
'''for max0 in maxs0[:2]:
    jj_center_name_type(save_path1+f'duzxx_table/graph/{max0}/', 'JJCC_hebin_gmjj_Fin副本.csv', '基金名称-基金类型.csv')
    jj_center_name_type(save_path1 + f'jjzxx_table/graph/{max0}/', 'JJCC_hebin_gmjj_Fin副本.csv', '基金名称-基金类型.csv')
    jj_center_name_type(save_path1 + f'jszxx_table/graph/{max0}/', 'JJCC_hebin_gmjj_Fin副本.csv', '基金名称-基金类型.csv')'''
'''for max0 in maxs0:
    cnm(folder_path2+f"graph/{max0}/", save_path1+f"communities_table/graph/{max0}/", number_to_rq)'''
'''df = pd.read_csv("table3(1)(1).csv", header=None, error_bad_lines=False, names=['公司名称', '公司代码', '基金名称', '基金类型',
                                                                                '基金规模', '基金经理', '成立日', '管理人',
                                                                                '基金评级', '备注一', '备注二', '备注三', '备注四'])
print(df)
df['基金类型'] = df['基金类型'].str.replace('基金类型：', '')
df['基金名称'] = df['基金名称'].str.extract(r'(.*) ', expand=False)
# df.rename(columns={'基金名称 基金代码': '基金名称'}, inplace=True)
df.to_csv("table3处理后.csv", index=False)'''
'''df0 = pd.read_csv("table.csv")
df0['基金类型'] = df0['基金类型'].str.replace('基金类型：', '')
df0['基金名称 基金代码'] = df0['基金名称 基金代码'].str.extract(r'(.*) ', expand=False)
df0.rename(columns={'基金名称 基金代码': '基金名称'}, inplace=True)
df0.to_csv("table处理后.csv", index=False)'''
'''df0 = pd.read_csv('table处理后.csv', usecols=['基金名称', '基金类型'])
df3 = pd.read_csv('table3处理后.csv', usecols=['基金名称', '基金类型'])
df = pd.concat([df3, df0], ignore_index=True)
df = df.drop_duplicates(subset='基金名称')
df.to_csv('基金名称-基金类型.csv', index=False)'''
'''df1 = pd.read_csv("JJCC_hebin_gmjj_Fin.csv", usecols=['基金名称', '股票名称', '基金代码', '股票代码'], low_memory=False)
df2 = pd.read_csv("jjst.csv", usecols=['jjid', 'stid', '开始时间', '结束时间', '规模(万股)'], low_memory=False)
df = df2.join(df1)
df.to_csv('JJCC_hebin_gmjj_Fin副本.csv', index=False)'''
# for max0 in maxs0[1:]:
#     jj_id2char(save_path1+f"communities_table/graph/{max0}/", 'JJCC_hebin_gmjj_Fin副本.csv', '基金名称-基金类型.csv')

# 股票
# for max0 in maxs0[:1]:
#     center(folder_path3 + f"graph/{max0}/", save_path2, max0)
# for max0 in maxs0[2:]:
#     cnm(folder_path3+f"graph/{max0}/", save_path2+f"communities_table/graph/{max0}/", number_to_rq)
# for max0 in maxs0[5:]:
#     st_id2char(save_path2+f"communities_table/graph/{max0}/", 'JJCC_hebin_gmjj_Fin副本2.csv', 'TRD_Co.csv')
for max0 in maxs0[:2]:
    st_center_name_ind(save_path2+f'duzxx_table/graph/{max0}/', 'JJCC_hebin_gmjj_Fin副本2.csv', 'TRD_Co.csv')
    st_center_name_ind(save_path2 + f'jjzxx_table/graph/{max0}/', 'JJCC_hebin_gmjj_Fin副本2.csv', 'TRD_Co.csv')
    st_center_name_ind(save_path2 + f'jszxx_table/graph/{max0}/', 'JJCC_hebin_gmjj_Fin副本2.csv', 'TRD_Co.csv')
