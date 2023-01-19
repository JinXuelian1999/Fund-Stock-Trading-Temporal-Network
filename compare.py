import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt


def bipartite_centrality(path, filename1, filename2, filename3, filename4):
    # 读取文件
    data1 = pd.read_csv(filename1, low_memory=False)
    data2 = pd.read_csv(filename2, low_memory=False)
    data3 = pd.read_csv(filename3, low_memory=False)
    data4 = pd.read_csv(filename4, low_memory=False)
    # 构造字典
    jj_ids = list(data1['jjid'])
    jj_names1 = list(data1['基金名称'])
    jj_id_to_name = dict(zip(jj_ids, jj_names1))
    jj_names2 = list(data2['基金名称'])
    jj_types = list(data2['enType'])
    jj_name_to_type = dict(zip(jj_names2, jj_types))
    jj_id_to_type = {u: jj_name_to_type[v] for u, v in jj_id_to_name.items() if v in jj_name_to_type.keys()}

    st_ids = list(data3['stid'])
    st_names = list(data3['股票名称'])
    st_id_to_name = dict(zip(st_ids, st_names))
    st_codes1 = list(data3['股票代码'].fillna(value=0).astype('int'))
    st_id_to_code = dict(zip(st_ids, st_codes1))
    st_code2 = list(data4['Stkcd'])
    st_ind = list(data4['enNindnme'])
    st_code_to_ind = dict(zip(st_code2, st_ind))
    st_id_to_ind = {u: st_code_to_ind[v] for u, v in st_id_to_code.items() if v in st_code_to_ind.keys()}

    # 合并
    id_to_name = {**jj_id_to_name, **st_id_to_name}
    id_to_type_or_ind = {**jj_id_to_type, **st_id_to_ind}

    files = os.listdir(path)
    files.sort()
    for file in files:
        print(file)
        data5 = pd.read_csv(path + file, low_memory=False)
        data5["name"] = data5["node"].map(id_to_name)
        data5["type or industry"] = data5["node"].map(id_to_type_or_ind)
        data5.to_csv(f"{path}{file}", index=False)


def center_fan_chart(path, t, figure_name, save_path):
    top_one = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        print(file)
        data = pd.read_csv(path + file, low_memory=False)
        data.dropna(how='any', inplace=True)
        data.index = range(data.shape[0])
        try:
            top_one.append(data.loc[0, t])
        except KeyError:
            continue
    print(top_one)
    print(len(top_one))
    counter = Counter(top_one)
    print(counter)
    # 设置画布的大小
    plt.figure(figsize=(10, 10))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 第一个传入的是我们需要计算的数据，
    plt.pie(counter.values(),
            # labels是传入标签的
            labels=counter.keys(),
            # 格式化输出百分比
            autopct='%.2f%%'
            )
    # 绘图的标题
    plt.title(figure_name, fontsize=27)
    # 存储图片
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + f"{figure_name}.png", format='png', dpi=100)
    plt.close()


folder_path1 = "G:/jj_st/bipartite/duzxx_table/"
folder_path2 = "G:/jj_st/bipartite/jjzxx_table/"
folder_path3 = "G:/jj_st/bipartite/jszxx_table/"
folder_path4 = "G:/jj_st/one_mode_graph/jj/duzxx_table/graph/0.1/"
folder_path5 = "G:/jj_st/one_mode_graph/jj/jjzxx_table/graph/0.1/"
folder_path6 = "G:/jj_st/one_mode_graph/jj/jszxx_table/graph/0.1/"
folder_path7 = "G:/jj_st/one_mode_graph/st/duzxx_table/graph/0.1/"
folder_path8 = "G:/jj_st/one_mode_graph/st/jjzxx_table/graph/0.1/"
folder_path9 = "G:/jj_st/one_mode_graph/st/jszxx_table/graph/0.1/"
if __name__ == "__main__":
    for folder_path in [folder_path1, folder_path2, folder_path3]:
        bipartite_centrality(folder_path, 'JJCC_hebin_gmjj_Fin副本.csv', 'En_JJ_Types.csv',
                             'JJCC_hebin_gmjj_Fin副本2.csv', 'En_TRD_Co.csv')
    # folder_paths = [folder_path1, folder_path2, folder_path3, folder_path4, folder_path5, folder_path6, folder_path7,
    #                 folder_path8, folder_path9]
    # names = ["基金股票网络-度中心性-饼图", "基金股票网络-接近中心性-饼图", "基金股票网络-介数中心性-饼图",
    #          "基金网络-度中心性-饼图", "基金网络-接近中心性-饼图", "基金网络-介数中心性-饼图",
    #          "股票网络-度中心性-饼图", "股票网络-接近中心性-饼图", "股票网络-介数中心性-饼图"]
    # save_paths = ["G:/jj_st/bipartite/", "G:/jj_st/bipartite/", "G:/jj_st/bipartite/",
    #               "G:/jj_st/one_mode_graph/jj/", "G:/jj_st/one_mode_graph/jj/", "G:/jj_st/one_mode_graph/jj/",
    #               "G:/jj_st/one_mode_graph/st/", "G:/jj_st/one_mode_graph/st/", "G:/jj_st/one_mode_graph/st/"]
    # types = ["type or industry"] * 3 + ["type"] * 3 + ["industry"] * 3
    #
    # for i in range(9):
    #     center_fan_chart(folder_paths[i], types[i], names[i], save_paths[i])

