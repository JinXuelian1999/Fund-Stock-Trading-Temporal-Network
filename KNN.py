from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from collections import Counter
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.feature_selection import RFE


def knn(path, year=1, dimension=None):
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
    # 分训练集、测试集
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    # 数据标准化
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.fit_transform(Xtest)
    # 降维
    # Xtrain = PCA(n_components=0.80).fit_transform(Xtrain)
    # Xtest = PCA(n_components=0.80).fit_transform(Xtest)
    # 调参
    # test = []
    # n_range = range(2, 21)
    # for i in n_range:
    #     clf = KNeighborsClassifier(n_neighbors=i)
    #     clf = clf.fit(Xtrain, Ytrain)
    #     score = clf.score(Xtest, Ytest)
    #     test.append(score)
    # # print(max(test), n_range[test.index(max(test))])
    # plt.plot(n_range, test, color="red", label="n_neighbors")
    # plt.legend()
    # plt.show()
    # """调参前"""
    # clf = KNeighborsClassifier().fit(Xtrain, Ytrain)
    # score = cross_val_score(clf, X, y, cv=10).mean()
    # print("调参前：", score)
    # """网格搜索"""
    # parameters = {
    #     "n_neighbors": [*range(1, 10)]
    # }
    #
    # clf = KNeighborsClassifier()
    # GS = GridSearchCV(clf, parameters, cv=10)
    # GS.fit(X, y)
    #
    # print("GS.best_params_", GS.best_params_)
    # print("GS.best_score_", GS.best_score_)
    # 训练模型及预测
    model = KNeighborsClassifier(n_neighbors=3).fit(Xtrain, Ytrain)
    pickle.dump(model, open("knn_预测1.dat", "wb"))
    # model = pickle.load(open("knn_预测.dat", "rb"))
    predict1 = model.predict(Xtrain)
    predict2 = model.predict(Xtest)
    # 评价模型
    print("KNN                 训练集：", end=' ')
    print("accuracy: %.3f" % accuracy_score(Ytrain, predict1), end=' ')
    print("precision: %.3f" % precision_score(Ytrain, predict1), end=' ')
    print("recall: %.3f" % recall_score(Ytrain, predict1), end=' ')
    print("F1: %.3f" % f1_score(Ytrain, predict1), end=' ')

    print("测试集：", end=' ')
    print("accuracy: %.3f" % accuracy_score(Ytest, predict2), end=' ')
    print("precision: %.3f" % precision_score(Ytest, predict2), end=' ')
    print("recall: %.3f" % recall_score(Ytest, predict2), end=' ')
    print("F1: %.3f" % f1_score(Ytest, predict2))


# if __name__ == "__main__":
#     p = "G:/jj_st/bipartite/"
#     knn(p)
