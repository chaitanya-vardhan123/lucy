t#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 23:33:46 2017

@author: Kanth
"""



# Gaussian Naive Bayes
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = pd.read_csv("/Users/kanth/Desktop/KNN.csv")
del dataset['id']

dataset.data1 = dataset.iloc[:,:1]
dataset.data1

dataset.data = pd.DataFrame(dataset.data1)

dataset.data.describe()

dataset.data2 = dataset.iloc[:,1:]
dataset.data2.describe()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.data2,dataset.data1, test_size=0.2, random_state=0)



# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
metrics.