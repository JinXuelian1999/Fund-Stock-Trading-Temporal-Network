import SVM
import Logistic
import naive_bayes
import decisiontree
import KNN
import Xgboost


if __name__ == "__main__":
    p = "G:/jj_st/bipartite/"

    # svm_dimensions = [4, 1, 4]
    # logistic_dimensions = [3, 1, 2]
    # bayes_dimensions = [2, 3, 4]
    # decisiontree_dimensions = [2, 3, 3]
    # knn_dimensions = [2, 1, 1]
    #
    # years = [3, 2, 1]       # 本季度结果、下一季度结果、下两季度结果

    # for _ in range(3):

    # print("SVM\t")
    SVM.svm(p)
    # print("Logistic\t")
    Logistic.logistic(p)
    # print("Naive Bayes\t")
    naive_bayes.bayes(p)
    # print("Decision Tree\t\t")
    decisiontree.decisiontree(p)
    # print("KNN\t\t")
    KNN.knn(p)
    Xgboost.xgbc(p)
    print('\n')
