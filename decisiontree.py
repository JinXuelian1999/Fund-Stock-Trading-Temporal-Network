from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.feature_selection import RFE


def decisiontree(path, year=1, dimension=None):
    # 读取数据
    data = pd.read_csv(path+'预测1.csv', usecols=["时间", "同配系数", "密度", "平均聚类系数", "同配系数1", "密度1", "平均聚类系数1",
                                                "股市"], index_col='时间')
    # data = pd.read_csv(path + "预测.csv", index_col="时间")
    data.dropna(how='any', inplace=True)
    # print(data.head())
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # print(X)
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
    # max_dep_range = range(1, 21)
    # for i in range(20):
    #     clf = tree.DecisionTreeClassifier(max_depth=i + 1
    #                                       , criterion="entropy"
    #                                       , random_state=30
    #                                       , splitter="random"
    #                                       )
    #     clf = clf.fit(Xtrain, Ytrain)
    #     score = clf.score(Xtest, Ytest)
    #     test.append(score)
    # # print(max(test), max_dep_range[test.index(max(test))])
    # plt.plot(range(1, 21), test, color="red", label="max_depth")
    # plt.legend()
    # plt.show()
    # """调参前"""
    # clf = tree.DecisionTreeClassifier(random_state=0, class_weight='balanced').fit(Xtrain, Ytrain)
    # score = cross_val_score(clf, X, y, cv=10).mean()
    # print("调参前：", score)
    # """网格搜索"""
    # parameters = {
    #     "splitter": ("best", "random"),
    #     "criterion": ("gini", "entropy"),
    #     "min_samples_leaf": [*range(1, 50, 1)],
    #     "min_impurity_decrease": [*np.linspace(0, 0.5, 20)],
    #     "max_depth": [*range(1, 10)]
    # }
    #
    # clf = tree.DecisionTreeClassifier(random_state=0, class_weight='balanced')
    # GS = GridSearchCV(clf, parameters, cv=10)
    # GS.fit(X, y)
    #
    # print("GS.best_params_", GS.best_params_)
    # print("GS.best_score_", GS.best_score_)
    # 训练模型及预测
    model = tree.DecisionTreeClassifier(random_state=0, class_weight='balanced', criterion='entropy', max_depth=2,
                                        min_impurity_decrease=0.0, min_samples_leaf=12, splitter='best').fit(Xtrain,
                                                                                                            Ytrain)
    pickle.dump(model, open("decisiontree_预测1.dat", "wb"))
    # print(model.feature_importances_)
    # model = pickle.load(open("decisiontree_预测.dat", "rb"))
    predict1 = model.predict(Xtrain)
    predict2 = model.predict(Xtest)
    # 评价模型
    print("Decision Tree       训练集：", end=' ')
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
#     decisiontree(p)
