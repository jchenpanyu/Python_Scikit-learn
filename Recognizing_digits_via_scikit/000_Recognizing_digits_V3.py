# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 17:55:54 2017

分類法/範例一: Recognizing hand-written digits

http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

這個範例用來展示scikit-learn 機器學習套件，如何用SVM演算法來達成手寫的數字辨識

利用 make_classification 建立模擬資料
利用 sklearn.datasets.load_digits() 來讀取內建資料庫
用線性的SVC來做分類，以8x8的影像之像素值來當作特徵(共64個特徵)
用 metrics.classification_report 來提供辨識報表

@author: vincchen
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import svm
from scipy import misc

# data set
img_list = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '2-1',  '2-2', '2-3', '2-4', '2-5'] 
data = np.zeros((len(img_list), 40*40))
for i, name in enumerate(img_list):
    img = Image.open(name+'.png')
    img = np.array(img.convert('L')).ravel()
    data[i] = img

label_list = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

# 產生SVC分類器
classifier = svm.SVC(gamma=0.00001)
# 用資料來訓練
classifier.fit(data, label_list)

#预测输入
img_test = Image.open('test-2.png')
img_test = np.array(img_test.convert('L'))
img_test = misc.imresize(img_test, size=(40, 40), interp='nearest')
plt.imshow(img_test, cmap=plt.cm.gray)
plt.show()

print classifier.predict(img_test.ravel())
