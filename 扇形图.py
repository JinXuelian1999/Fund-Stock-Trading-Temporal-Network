import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter


def jj_communities_fan(folder_path, save_path):
    files = os.listdir(folder_path)
    files.sort()
    for file in files[:1]:
        print(file)
        # 读取文件
        data = pd.read_csv(folder_path + file)
        figure_name = re.sub('\\.csv', '', file)
        data['基金类型'] = data['基金类型'].map(eval)
        communities = list(data['community'].unique())
        data.set_index('community', inplace=True)  # 将社团设置为索引
        for com in communities:
            member_rate = Counter(data['基金类型'].loc[com])
            # 设置画布的大小
            plt.figure(figsize=(10, 10))
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            # 第一个传入的是我们需要计算的数据，
            plt.pie(member_rate.values(),
                    # labels是传入标签的
                    labels=member_rate.keys(),
                    # 格式化输出百分比
                    autopct='%.2f%%'
                    )
            # 绘图的标题
            plt.title(figure_name+f' 社团{com}', fontsize=27)
            # 存储图片
            print(save_path)
            if not os.path.exists(save_path+f"{figure_name}/"):
                os.mkdir(save_path+f"{figure_name}/")
            plt.savefig(save_path+f"{figure_name}/社团{com}.png", format='png', dpi=100)
            plt.close()


def st_communities_fan(folder_path, save_path):
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        print(file)
        # 读取文件
        data = pd.read_csv(folder_path + file)
        figure_name = re.sub('\\.csv', '', file)
        data['股票行业'] = data['股票行业'].map(eval)
        communities = list(data['community'].unique())
        data.set_index('community', inplace=True)  # 将社团设置为索引
        for com in communities:
            member_rate = Counter(data['股票行业'].loc[com])
            # print(member_rate)
            # 设置画布的大小
            plt.figure(figsize=(10, 10))
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            # 第一个传入的是我们需要计算的数据，
            plt.pie(member_rate.values(),
                    # labels是传入标签的
                    labels=member_rate.keys(),
                    # 格式化输出百分比
                    autopct='%.2f%%'
                    )
            # 绘图的标题
            plt.title(figure_name+f' 社团{com}', fontsize=27)
            # 存储图片
            # print(save_path)
            if not os.path.exists(save_path+f"{figure_name}/"):
                os.mkdir(save_path+f"{figure_name}/")
            plt.savefig(save_path+f"{figure_name}/社团{com}.png", format='png', dpi=100)
            plt.close()


if __name__ == "__main__":
    maxs0 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    maxs1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
    path1 = 'G:/jj_st/one_mode_graph/jj/communities_table/graph/'
    path2 = 'G:/jj_st/one_mode_graph/jj/community_fan_charts/graph/'
    path3 = 'G:/jj_st/one_mode_graph/st/communities_table/graph/'
    path4 = 'G:/jj_st/one_mode_graph/st/community_fan_charts/graph/'
    # for max0 in maxs0:
    #     jj_communities_fan(f"{path1}{max0}/", f"{path2}{max0}/")
    for max0 in maxs0:
        st_communities_fan(f"{path3}{max0}/", f"{path4}{max0}/")

