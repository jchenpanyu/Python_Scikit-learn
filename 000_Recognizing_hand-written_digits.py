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
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

for key,value in digits.items() :
    try:
        print (key,value.shape)
    except:
        print (key)

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    
n_samples = len(digits.images)

# 資料攤平:1797 x 8 x 8 -> 1797 x 64
# 這裏的-1代表自動計算，相當於 (n_samples, 64)
data = digits.images.reshape((n_samples, -1))

# 產生SVC分類器
classifier = svm.SVC(gamma=0.001)

# 用前半部份的資料來訓練
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

expected = digits.target[n_samples / 2:]

#利用後半部份的資料來測試分類器，共 899筆資料
predicted = classifier.predict(data[n_samples / 2:])

# 觀察 expected 及 predicted 矩陣中之前10個變數可以得到:
print 'first 10 expected result:', expected[:10]
print 'first 10 predicted result:', predicted[:10]

# 在判斷準確度方面，我們可以使用一個名為「混淆矩陣」(Confusion matrix)的方式來統計
"""
使用sklearn中之metrics物件，metrics.confusion_matrix(真實資料:899, 預測資料:899)可以列出下面矩陣。
此矩陣對角線左上方第一個數字 87，代表實際為0且預測為0的總數有87個，
同一列(row)第五個元素則代表，實際為0但判斷為4的資料個數為1個。
"""
print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(expected, predicted))

# 我們可以利用以下的程式碼將混淆矩陣圖示出來。由圖示可以看出，實際為3時，有數次誤判為5,7,8。
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    import numpy as np
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(digits.target_names))
    plt.xticks(tick_marks, digits.target_names, rotation=45)
    plt.yticks(tick_marks, digits.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(metrics.confusion_matrix(expected, predicted))

"""
以手寫影像3為例，我們可以用四個數字來探討判斷的精準度。

True Positive(TP,真陽):實際為3且判斷為3，共79個
False Positive(FP,偽陽):判斷為3但判斷錯誤，共2個
False Negative(FN,偽陰):實際為3但判斷錯誤，共12個
True Negative(TN,真陰):實際不為3，且判斷正確。也就是其餘899-79-2-12=885個
而在機器學習理論中，我們通常用以下precision, recall, f1-score來探討精確度。以手寫影像3為例。

precision = TP/(TP+FP) = 79/81 = 0.98
判斷為3且實際為3的比例為0.98
recall = TP/(TP+FN) = 79/91 = 0.87
實際為3且判斷為3的比例為0.87
f1-score 則為以上兩者之「harmonic mean 調和平均數」
f1-score= 2 x precision x recall/(recision + recall) = 0.92
"""

# metrics物件裏也提供了方便的函式metrics.classification_report(expected, predicted)計算以上統計數據。
print("Classification report for classifier %s:\n%s\n"
    % (classifier, metrics.classification_report(expected, predicted)))


# 最後，用以下的程式碼可以觀察測試影像以及預測(分類)結果得對應關係。
images_and_predictions = list(
                        zip(digits.images[n_samples / 2:], predicted))
plt.figure()
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()