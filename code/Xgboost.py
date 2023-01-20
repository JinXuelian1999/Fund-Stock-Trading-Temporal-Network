import xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


def xgbc(path):
    # 读取数据
    data = pd.read_csv(path + '预测.csv', usecols=["time", "DS", "ACC", "DM",  "Bull or Bear market-next quarter"],
                       index_col='time')
    data.dropna(how='any', inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # # 负正样本比例
    # classes = Counter(y)
    # # print(classes)
    # rate = classes[0.0] / classes[1.0]
    # print(rate)
    # """调参前"""
    # clf = XGBClassifier(random_state=0, scale_pos_weight=1)
    # score = cross_val_score(clf, X, y, cv=10).mean()
    # print("调参前：", score)
    # """网格搜索"""
    # parameters = {
    #     'n_estimators': range(1, 10),
    #     'learning_rate': [*np.linspace(0, 0.5, 20)],
    #     'max_depth': range(1, 8)
    # }
    #
    # clf = XGBClassifier(random_state=0, scale_pos_weight=1)
    # GS = GridSearchCV(clf, parameters, cv=10)
    # GS.fit(X, y)
    #
    # print("GS.best_params_", GS.best_params_)
    # print("GS.best_score_", GS.best_score_)
    # 调参max_depth
    # dfull = xgb.DMatrix(X, y)
    # param1 = {"random_state": 0, "learning_rate": 0.3684210526315789, "max_depth": 6, "objective": "binary:logistic",
    #           "scale_pos_weight": rate}
    # param2 = {"random_state": 0, "learning_rate": 0.3684210526315789, "max_depth": 1, "objective": "binary:logistic",
    #           "scale_pos_weight": rate}
    # num_round = 10
    # n_fold = 10
    # cvresult1 = xgb.cv(param1, dfull, num_round, n_fold, metrics=("error"))
    # cvresult2 = xgb.cv(param2, dfull, num_round, n_fold, metrics=("error"))
    # # cvresult1
    #
    # plt.figure(figsize=(20, 5))
    # plt.grid()
    # plt.plot(range(1, 11), cvresult1.iloc[:, 0], c="red", label="train, max_depth=6")
    # plt.plot(range(1, 11), cvresult1.iloc[:, 2], c="orange", label="test, max_depth=6")
    # plt.plot(range(1, 11), cvresult2.iloc[:, 0], c="green", label="train, max_depth=1")
    # plt.plot(range(1, 11), cvresult2.iloc[:, 2], c="blue", label="test, max_depth=1")
    # plt.legend()
    # plt.show()
    # 建模
    model = XGBClassifier(random_state=0, scale_pos_weight=1, learning_rate=0.42105263157894735, n_estimators=6,
                          max_depth=6)
    # model.fit(X_train, Y_train)
    # pickle.dump(model, open("xgboost_预测.dat", "wb"))
    # # model = pickle.load(open("xgboost_预测.dat", "rb"))
    # # 预测
    # predict1 = model.predict(X_train)
    # predict2 = model.predict(X_test)
    #
    # # 评价模型
    # print("Xgboost             训练集：", end=' ')
    # print("accuracy: %.3f" % accuracy_score(Y_train, predict1), end=' ')
    # print("precision: %.3f" % precision_score(Y_train, predict1), end=' ')
    # print("recall: %.3f" % recall_score(Y_train, predict1), end=' ')
    # print("F1: %.3f" % f1_score(Y_train, predict1), end=' ')
    #
    # print("测试集：", end=' ')
    print("accuracy: %.4f" % cross_val_score(model, X, y, cv=10, scoring='accuracy').mean(), end=' ')
    print("precision: %.4f" % cross_val_score(model, X, y, cv=10, scoring='precision').mean(), end=' ')
    print("recall: %.4f" % cross_val_score(model, X, y, cv=10, scoring='recall').mean(), end=' ')
    print("F1: %.4f" % cross_val_score(model, X, y, cv=10, scoring='f1').mean())


if __name__ == "__main__":
    p = "G:/jj_st/bipartite/"
    xgbc(p)
