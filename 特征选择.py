from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
import itertools
import csv


path = "G:/jj_st/bipartite/"
# 读取数据
data = pd.read_csv(path+'预测.csv', usecols=["time", "DM", "ADF", "ADS", "AST", "DS", "ACC",
                                           "Bull or Bear market-next quarter"], index_col='time')
# data = pd.read_csv(path + "预测1.csv", index_col="时间")
data.dropna(how='any', inplace=True)

features = ["DM", "ADF", "ADS", "AST", "DS", "ACC"]
fs = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

file = open("jjst_feature_selection.csv", mode='w', encoding='utf-8', newline='')
csvwriter = csv.writer(file)
for i in range(1, 7):
    for j in itertools.combinations(features, i):
        df = data[list(j) + ["Bull or Bear market-next quarter"]]
        # print(df)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        clf = XGBC(random_state=0)
        accuracy_score = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
        precision_score = cross_val_score(clf, X, y, cv=10, scoring='precision').mean()
        recall_score = cross_val_score(clf, X, y, cv=10, scoring='recall').mean()
        f1_score = cross_val_score(clf, X, y, cv=10, scoring='f1').mean()
        fs.append(list(j))
        accuracy_scores.append(accuracy_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        f1_scores.append(f1_score)
        csvwriter.writerow(['&'.join(list(j)), format(accuracy_score, '.4f'), format(precision_score, '.4f'),
                            format(recall_score, '.4f'), format(f1_score, '.4f')])
        print(f"特征：{list(j)}, accuracy: {accuracy_score}, precision: {precision_score}, recall: {recall_score}, "
              f"f1: {f1_score}")
file.close()

print(max(accuracy_scores), fs[accuracy_scores.index(max(accuracy_scores))])
print(max(precision_scores), fs[precision_scores.index(max(precision_scores))])
print(max(recall_scores), fs[recall_scores.index(max(recall_scores))])
print(max(f1_scores), fs[f1_scores.index(max(f1_scores))])
# # 包装法筛选特征
# score = []
# clf = XGBC(random_state=0)
# for i in range(1, 6, 1):
#     X_wrapper = RFE(clf, n_features_to_select=i, step=1).fit_transform(X, y)
#     once = cross_val_score(clf, X_wrapper, y, cv=5).mean()
#     score.append(once)
# plt.figure(figsize=(20, 5))
# plt.plot(range(1, 6, 1), score)
# plt.xticks(range(1, 6, 1))
# plt.show()
#
# selector = RFE(clf, n_features_to_select=score.index(max(score))+1, step=1).fit(X, y)
# print("feature number: ", selector.support_.sum())
# print("selected features: ", selector.support_)
# x_wrapper = selector.transform(X)
