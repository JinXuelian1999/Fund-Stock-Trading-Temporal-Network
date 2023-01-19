import pandas as pd
import os
import numpy as np
import re


def convert(file_path, file, save_path):
    """将csv文件转换成多级表头形式"""
    data = pd.read_csv(file_path+file)
    data['members'] = data['members'].map(eval)
    data['基金名称'] = data['基金名称'].map(eval)
    data['基金类型'] = data['基金类型'].map(eval)
    columns = list(data['community'])
    data.set_index('community', inplace=True)
    com = []
    for column in columns:
        # content = np.array([data.loc[column, 'members'], data.loc[column, '基金名称'], data.loc[column, '基金类型']])
        one_com = pd.DataFrame({'基金ID': data.loc[column, 'members'],
                                '基金名称': data.loc[column, '基金名称'],
                                '基金类型': data.loc[column, '基金类型']})
        com.append(one_com)
    keys = ['社团'+str(x) for x in columns]
    result = pd.concat(com, keys=keys, axis=1)
    # print(result)
    file_name = re.sub('\\.csv', '.xlsx', file)
    # print(file_name)
    result.to_excel(save_path+file_name)


maxs0 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
maxs1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
path = 'G:/jj_st/one_mode_graph/jj/communities_table/graph/'
for max0 in maxs0[3:]:
    files = os.listdir(path+f"{max0}/")
    # print(files)
    for f in files:
        print(f)
        convert(path+f"{max0}/", f, path+'社团ID-基金名称-基金类型/'+f"{max0}/")
