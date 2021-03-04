# coding:UTF-8
import numpy as np
import scipy.io as scio
# Import svm model
from sklearn import svm
from sklearn import metrics
import os
from EmailFeatures import get_train_test_dataset

# load training and testing dataset
path = os.getcwd() + '\\dataset'
ratio = 0.6
x_train, y_train, x_test, y_test = get_train_test_dataset(path, ratio)
# 适应模型函数参数要求 转化为一维
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
# Create a svm Classifier
clf = svm.SVC(kernel='linear', class_weight='balanced')  # Linear Kernel

# Train the model using the training sets
clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(x_test)

# Evaluate the model
# Import scikit-learn metrics module for accuracy calculation
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))