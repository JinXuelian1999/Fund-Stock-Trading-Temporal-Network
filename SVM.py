from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from collections import Counter
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle


def svm(path, year=1, dimension=None):
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
    #X_dr = PCA(n_components=dimension).fit_transform(X)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    # 数据标准化
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.fit_transform(Xtest)
    # 降维
    # Xtrain = PCA(n_components=0.80).fit_transform(Xtrain)
    # Xtest = PCA(n_components=0.80).fit_transform(Xtest)
    # 调参
    # score1 = []
    # gamma_range = np.logspace(-10, 1, 50)  # 返回在对数刻度上均匀间隔的数字
    # for i in gamma_range:
    #     clf = SVC(kernel="rbf", gamma=i, cache_size=5000).fit(Xtrain, Ytrain)
    #     score1.append(clf.score(Xtest, Ytest))
    #
    # # print(max(score1), gamma_range[score1.index(max(score1))])
    # plt.plot(gamma_range, score1)
    # plt.xlabel("gamma")
    # plt.ylabel("score")
    # plt.show()
    #
    # score2 = []
    # C_range = np.linspace(0.01, 30, 50)
    # for i in C_range:
    #     clf = SVC(kernel="rbf", C=i, gamma=gamma_range[score1.index(max(score1))], cache_size=5000).fit(Xtrain, Ytrain)
    #     score2.append(clf.score(Xtest, Ytest))
    #
    # # print(max(score2), C_range[score2.index(max(score2))])
    # plt.plot(C_range, score2)
    # plt.xlabel("C")
    # plt.ylabel("score")
    # plt.show()
    # """调参前"""
    # clf = SVC(kernel="rbf", random_state=0, class_weight='balanced', cache_size=5000).fit(Xtrain, Ytrain)
    # score = cross_val_score(clf, X, y, cv=10).mean()
    # print("调参前：", score)
    # """网格搜索"""
    # parameters = {
    #     "C": [*np.linspace(0.01, 30, 50)],
    #     "gamma": [*np.logspace(-10, 1, 50)],
    # }
    # clf = SVC(kernel="rbf", random_state=0, class_weight='balanced', cache_size=5000)
    # GS = GridSearchCV(clf, parameters, cv=10)
    # GS.fit(X, y)
    #
    # print("GS.best_params_", GS.best_params_)
    # print("GS.best_score_", GS.best_score_)
    # 训练模型及预测
    model = SVC(kernel="rbf", C=8.57857142857143, gamma="auto",
                cache_size=5000, random_state=0, class_weight='balanced').fit(Xtrain, Ytrain)
    pickle.dump(model, open("svm_预测1.dat", "wb"))
    # model = pickle.load(open("svm_预测.dat", "rb"))
    predict1 = model.predict(Xtrain)
    predict2 = model.predict(Xtest)
    # 评价模型
    print("SVM                 训练集：", end=' ')
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
#     svm(p)

