#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 14:57:43 2017

@author: kanth
"""

import pandas as pd
from sklearn import datasets
from sklearn import metrics

# load the iris datasets
dataset = pd.read_csv("/Users/kanth/Documents/Certified Data Science Program Files/Machine Learning Files/Support Vector Machines/letterdata.csv")

dataset.data1 = dataset.iloc[:,:1]
dataset.data2 = dataset.iloc[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.data2,dataset.data1, test_size=0.2, random_state=0)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(5,5,5))
model.fit(X_train, y_train)


print(model)

# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
