from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV


def bayes(path, year=1, dimension=None):
    # 读取数据
    data = pd.read_csv(path + '预测1.csv', usecols=["时间", "同配系数", "密度", "平均聚类系数", "同配系数1", "密度1",
                                                  "平均聚类系数1", "股市"], index_col='时间')
    # data = pd.read_csv(path + "预测1.csv", index_col="时间")
    data.dropna(how='any', inplace=True)
    # print(data.head())
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # print(X)
    # print(y)
    # 数据降维
    # coms_n = list(range(1, 6))
    # for _ in coms_n:
    # X_dr = PCA(n_components=dimension).fit_transform(X)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)
    # 数据标准化
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.fit_transform(Xtest)
    # 降维
    # Xtrain = PCA(n_components=0.80).fit_transform(Xtrain)
    # Xtest = PCA(n_components=0.80).fit_transform(Xtest)
    # 建模、训练模型及预测
    gnb = GaussianNB().fit(Xtrain, Ytrain)  # 查看分数
    Y_pred1 = gnb.predict(Xtrain)
    Y_pred2 = gnb.predict(Xtest)
    # 评价模型
    print("Naive Bayes         训练集：", end=' ')
    print("accuracy: %.3f" % accuracy_score(Ytrain, Y_pred1), end=' ')
    print("precision: %.3f" % precision_score(Ytrain, Y_pred1), end=' ')
    print("recall: %.3f" % recall_score(Ytrain, Y_pred1), end=' ')
    print("F1: %.3f" % f1_score(Ytrain, Y_pred1), end=' ')

    print("测试集：", end=' ')
    print("accuracy: %.3f" % accuracy_score(Ytest, Y_pred2), end=' ')
    print("precision: %.3f" % precision_score(Ytest, Y_pred2), end=' ')
    print("recall: %.3f" % recall_score(Ytest, Y_pred2), end=' ')
    print("F1: %.3f" % f1_score(Ytest, Y_pred2))


# if __name__ == "__main__":
#     p = "G:/jj_st/bipartite/"
#     bayes(p, 2, 3)

