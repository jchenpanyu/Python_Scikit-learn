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
from scipy import misc
from sklearn import datasets, svm

# data set
img_list= ['1-1', '1-2', '1-3', '1-4', '1-5', '2-1',  '2-2', '2-3', '2-4', '2-5]













# The digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = np.zeros((1797, 16, 16))

# upsample training image to 16x16
for i in np.arange(n_samples):
     data[i] = misc.imresize(digits.images[i], size=(16, 16), interp='nearest')

# 資料攤平:1797 x 16 x 16 -> 1797 x 256
# 這裏的-1代表自動計算，相當於 (n_samples, 64)
data = data.reshape((n_samples, -1))

# 產生SVC分類器
classifier = svm.SVC(gamma=0.001)
# 用資料來訓練
classifier.fit(data[:], digits.target[:])

#预测输入
img = Image.open('1_16x16.png')
img_gray = 255 - np.array(img.convert('L'))
plt.imshow(img_gray, cmap=plt.cm.gray_r)
plt.show()

predicted = classifier.predict(img_gray.ravel())
print predicted