# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:23:41 2018

@author: kumar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


email_data =pd.read_csv('NB.csv',encoding='ISO-8859-1')
email_data.shape
email_data.info()
email_data['type'].value_counts()
email_data.loc[email_data['type']=='spam','type'] = 0
email_data.loc[email_data['type']=='ham','type'] =1
email_data.info()
email_data['type'].value_counts()
email_data['type'] = email_data['type'].astype('int')
#or  email_data['type'] = pd.to_numeric(email_data['type'])
email_data.info()

cv =TfidfVectorizer(stop_words ='english')
#words = cv.fit_transform(['Hello Halya How Are You','Hello Halya How Are You today,Lovey gal'])
#words.toarray()
#words = cv.get_feature_names() # removes, are, you scuh kind of stop words

#Data Cleaning

# changing to lower case
email_data['text'] = email_data['text'].str.lower()

#Function remove all numbers and special characters
def reg_exp(text):
    return re.sub("[^a-zA-Z\t\n\r\f\v]+",' ',text)

email_data['text'] =email_data['text'].apply(reg_exp)


x_train,x_test,y_train,y_test = train_test_split(email_data['text'],email_data['type'],test_size =0.2,random_state =5)
x_traincv = cv.fit_transform(x_train)
x_testcv = cv.transform(x_test)
a = x_traincv.toarray()# Document Term Metrics (DTM)
b =  x_testcv.toarray()
words = cv.get_feature_names()# corpus (words list)
len(words) # 7486 ,decresed compared to count vectorizer, removes are, you etc
len(a)#4447


#stemming the words
def doc_stem(text):
    ps =PorterStemmer()
    return ps.stem(text)
words =pd.DataFrame(words)
words[0] = words[0].apply(doc_stem)
# Checking 4000 row , document values and its text 
doc_4000 = x_train.iloc[4000]
doc_4000_metrics = a[4000]
doc_4000_inv = cv.inverse_transform(a[4000])

 
#Building Model using Naive Bayes

mnb =MultinomialNB()
email_train = mnb.fit (x_traincv,y_train)
email_train_predict = mnb.predict(x_traincv)

#Training Accuracy
metrics.accuracy_score(y_train,email_train_predict) #97

#Testing  Accuracy
email_test_predict = mnb.predict(x_testcv)
metrics.accuracy_score(y_test,email_test_predict) #96

import nltk
nltk.download('wordnet')