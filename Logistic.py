import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pickle


def logistic(path, year=1, dimension=None):
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
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    # 数据标准化
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.fit_transform(Xtest)
    # 降维
    # Xtrain = PCA(n_components=0.80).fit_transform(Xtrain)
    # Xtest = PCA(n_components=0.80).fit_transform(Xtest)
    # 调参
    # l = []
    # ltest = []
    # c_range = np.linspace(0.05, 1.5, 19)
    # for i in np.linspace(0.05, 1.5, 19):
    #     lrl = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
    #     lrl = lrl.fit(Xtrain, Ytrain)
    #     l.append(accuracy_score(lrl.predict(Xtrain), Ytrain))
    #     ltest.append(accuracy_score(lrl.predict(Xtest), Ytest))
    # graph = [l, ltest]
    # color = ["green", "lightgreen"]
    # label = ["L", "Ltest"]
    #
    # # print("L:", max(ltest), c_range[ltest.index(max(ltest))])
    #
    # plt.figure(figsize=(6, 6))
    # for i in range(len(graph)):
    #     plt.plot(np.linspace(0.05, 1.5, 19), graph[i], color[i], label=label[i])
    # plt.legend(loc=4)
    # plt.show()
    # """调参前"""
    # clf = LR(penalty="l2", solver="liblinear", random_state=0, class_weight='balanced',
    #          max_iter=1000).fit(Xtrain, Ytrain)
    # score = cross_val_score(clf, X, y, cv=10).mean()
    # print("调参前：", score)
    # """网格搜索"""
    # parameters = {
    #     "C": [*np.linspace(0.05, 1.5, 19)]
    # }
    #
    # clf = LR(penalty="l2", solver="liblinear", random_state=0, class_weight='balanced',
    #          max_iter=1000)
    # GS = GridSearchCV(clf, parameters, cv=10)
    # GS.fit(X, y)
    #
    # print("GS.best_params_", GS.best_params_)
    # print("GS.best_score_", GS.best_score_)
    # 训练模型及预测
    model = LR(penalty="l2", solver="liblinear", C=0.29166666666666663, random_state=0, class_weight='balanced',
               max_iter=1000).fit(Xtrain, Ytrain)
    pickle.dump(model, open("logistic_预测1.dat", "wb"))
    # model = pickle.load(open("logistic_预测.dat", "rb"))
    predict1 = model.predict(Xtrain)
    predict2 = model.predict(Xtest)
    # 评价模型
    print("Logistic            训练集：", end=' ')
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
#     logistic(p)
