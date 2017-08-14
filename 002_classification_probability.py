# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 17:55:54 2017

分類法/範例三: Plot classification probability

這個範例的主要目的

使用iris 鳶尾花資料集
測試不同分類器對於涵蓋特定範圍之資料集，分類為那一種鳶尾花的機率
例如：sepal length 為 4cm 而 sepal width 為 3cm時被分類為 versicolor的機率

@author: vincchen
"""

"""
(一)資料匯入及描述
首先先匯入iris 鳶尾花資料集，使用iris = datasets.load_iris()將資料存入
準備X (特徵資料) 以及 y (目標資料)，僅使用兩個特徵方便視覺呈現
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, 0:2]  # 僅使用前兩個特徵，方便視覺化呈現
y = iris.target

n_features = X.shape[1]

# iris為一個dict型別資料，我們可以用以下指令來看一下資料的內容。
for key,value in iris.items() :
    try:
        print (key,value.shape)
    except:
        print (key)


"""
(二) 分類器的選擇
這個範例選擇了四種分類器，存入一個dict資料中，分別為：

L1 logistic
L2 logistic (OvR)
Linear SVC
L2 logistic (Multinomial)
其中LogisticRegression 並不適合拿來做多目標的分類器，我們可以用結果圖的分類機率來觀察。
"""
C = 1.0
# Create different classifiers. The logistic regression cannot do multiclass out of the box.
classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
               'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                                 random_state=0),
               'L2 logistic (Multinomial)': LogisticRegression(
                C=C, solver='lbfgs', multi_class='multinomial'
                )}

n_classifiers = len(classifiers)


"""
產生一個網格矩陣，其中xx,yy分別代表著iris資料集的第一及第二個特徵。
xx 是3~9之間的100個連續數字，而yy是1~5之間的100個連續數字。
用np.meshgrid(xx,yy)及np.c_產生出Xfull特徵矩陣，10,000筆資料包含了兩個特徵的所有排列組合。
"""
plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=.2, top=.95)

xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]


"""
(三) 測試分類器以及畫出機率分佈圖的選擇
接下來的動作
用迴圈輪過所有的分類器，並計算顯示分類成功率
將Xfull(10000x2矩陣)傳入 classifier.predict_proba()得到probas(10000x3矩陣)。
這裏的probas矩陣是10000種不同的特徵排列組合所形成的數據，被分類到三種iris 鳶尾花的可能性。
利用reshape((100,100))將10000筆資料排列成二維矩陣，並將機率用影像的方式呈現出來
"""
for index, (name, classifier) in enumerate(classifiers.items()):
    #訓練並計算分類成功率
    #然而此範例訓練跟測試用相同資料集，並不符合實際狀況。
    #建議採用cross_validation的方式才能較正確評估
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
    print("classif_rate for %s : %f " % (name, classif_rate))

    # View probabilities=
    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(3, 9, 1, 5), origin='lower')
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred == k)
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='k')

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

plt.show()



