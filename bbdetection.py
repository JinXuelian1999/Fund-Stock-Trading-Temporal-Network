from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from io import BytesIO
import PIL.Image as Image
import requests
import time
import csv
import rpy2
import pandas as pd
from rpy2.robjects import pandas2ri
import numpy as np


def get_data():
    url = 'http://19.push2his.eastmoney.com/api/qt/stock/kline/get'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.50"
                      "60.114 Safari/537.36 Edg/103.0.1264.62",
        "Referer": 'http://quote.eastmoney.com/'
    }
    params = {
        "secid": '1.000300',
        "ut": 'fa5fd1943c7b386f172d6893dbfba10b',
        "fields1": 'f1,f2,f3,f4,f5,f6',
        "fields2": 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        "klt": '103',
        "fqt": '1',
        "beg": '0',
        "end": '20500101',
        "smplmt": '1039',
        "lmt": '1000000',
        "_": '1658041590274'
    }
    resp = requests.get(url, params=params, headers=headers)
    # print(resp.url)
    print(resp.json())
    result = resp.json()
    klines = result['data']['klines']
    f = open("沪深300.csv", mode='a', encoding='utf-8', newline='')
    csvwriter = csv.writer(f)
    csvwriter.writerow(["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅",
                        "涨跌额", "换手率"])
    for kline in klines:
        csvwriter.writerow(kline.split(','))


def month2quarter(x):
    if x <= 3:
        return 1
    elif x <= 6:
        return 2
    elif x <= 9:
        return 3
    else:
        return 4


def bb_detection(hs300m):
    # 准备工作
    r = robjects.r
    r.rm(list=r.ls(all=True))
    bbdetection = importr('bbdetection')
    zoo = importr('zoo')
    xtable = importr('xtable')
    ggplot2 = importr('ggplot2')
    methods = importr('methods')
    pandas2ri.activate()
    robjects.globalenv['hs300m'] = hs300m
    # print(robjects.globalenv['hs300m'])
    as_vector = r['as.vector']
    prices = as_vector(r.coredata(hs300m))
    # print(prices)
    dates = r.index(hs300m)
    # print(dates)

    bbdetection.setpar_dating_alg(8, 6, 4, 16, 20)
    bull = bbdetection.run_dating_alg(prices)

    # 结果可视化
    p = bbdetection.bb_plot(prices, bull, dates, "沪深300")

    # 打印出熊市、牛市的时间
    as_yearmon = r['as.yearmon']
    dates = as_yearmon(dates)
    # sys_locale = r['Sys.setlocale']
    robjects.globalenv['Sys.setlocale'] = robjects.StrVector(["LC_TIME", "English"])
    frame0 = bbdetection.bb_dating_states(prices, bull, dates)
    print(frame0)

    # 打印出对牛熊市的总结
    frame1 = bbdetection.bb_summary_stat(prices, bull)
    print(frame1)

    return list(bull)


if __name__ == "__main__":
    # get_data()
    # print(rpy2.__version__)
    df = pd.read_csv("沪深300.csv", usecols=["时间", "收盘"])
    data = pd.Series(dict(zip(df["时间"], df["收盘"])))
    data.name = "hs300"
    # print(data)
    bb = bb_detection(data)
    result = pd.DataFrame({"时间": pd.to_datetime(df["时间"], format="%Y-%m-%d"), "股市": bb})
    result["股市"] = result["股市"].astype("int")
    result["年份"] = result["时间"].dt.year
    result["月份"] = result["时间"].dt.month
    result["季度"] = result["月份"].map(month2quarter)
    # result.to_csv("bbdetection.csv", index=False)

    # df1 = pd.read_csv("bbdetection.csv")
    # grouped = df1["股市"].groupby([df1["年份"], df1["季度"]]).agg(lambda x: np.mean(x.mode())).reset_index()
    # grouped.to_csv("牛熊市结果-季度.csv", index=False)
