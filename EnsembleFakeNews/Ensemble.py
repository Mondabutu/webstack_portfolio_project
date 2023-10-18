# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 09:50:59 2023

@author: Emi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,precision_score,recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
from wordcloud import WordCloud

df1 = pd.read_csv('News_dataset/news.csv')


df1['titletext'] = df1.title + ' ' + df1.text
df1.drop(['subject','title','text'], axis=1, inplace=True)
print('News',df1.head())

df2 = pd.read_csv('News_dataset/True.csv')
df2.drop(['subject','date'], axis=1, inplace=True)
df2['label'] = 'REAL'


df2['titletext'] = df2.title + ' ' + df2.text
df2.drop(['title','text'], axis=1, inplace=True)
print('True',df2.head())

df3 = pd.read_csv('News_dataset/Fake.csv')

df3.drop(['subject','date'], axis=1, inplace=True)
df3['label'] = 'FAKE'


df3['titletext'] =df3.title + ' ' + df3.text
df3.drop(['title','text'], axis=1, inplace=True)

print('Fake', df3.head())

df1.isnull().sum()
df2.isnull().sum()

df3.isnull().sum()

news = pd.concat([df1,df2,df3])

news.describe()
print('News Header',news.head())
print(news.sample(10))


print('News shape',news.shape)
X=news.titletext

X.shape

Y=news.label

Y.shape
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train.astype('U').values)
tfidf_test = tfidf_vectorizer.transform(x_test.astype('U').values)
#Passive Agressive Classifier
PAC=PassiveAggressiveClassifier(max_iter=4000)
PAC=PassiveAggressiveClassifier(max_iter=4000)
PAC.fit(tfidf_train,y_train.astype('U').values)
rfc_pred=PAC.predict(tfidf_test)


print('Passive Agressive Classifier Performance')
score2=accuracy_score(y_test.astype('U').values,rfc_pred)
print(f'Passive Agressive Classifier Accuracy: {round(score2*100,2)}%')
F1_score=f1_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Passive Agressive Classifier F1-Score: {round(F1_score*100,2)}%')
Precision=precision_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Passive Agressive Classifier Precision: {round(Precision*100,2)}%')
recall = recall_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Passive Agressive Classifier Recall: {round(recall*100,2)}%')

print('')

 


#Import svm model
from sklearn import svm

#Create a svm Classifier
SVMclf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
SVMclf.fit(tfidf_train,y_train.astype('U').values)

#Predict the response for test dataset


rfc_pred = SVMclf.predict(tfidf_test)
print('Support Vector Machine Performance')
score2=accuracy_score(y_test.astype('U').values,rfc_pred)
print(f'Support Vector Machine Accuracy: {round(score2*100,2)}%')
F1_score=f1_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Support Vector Machine F1-Score: {round(F1_score*100,2)}%')
Precision=precision_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Support Vector Machine Precision: {round(Precision*100,2)}%')
recall = recall_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Support Vector Machine Recall: {round(recall*100,2)}%')



from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

#Create a svm Classifier
Dclf = DecisionTreeClassifier()# Decision Tree

#Train the model using the training sets
Dclf.fit(tfidf_train,y_train.astype('U').values)

#Predict the response for test dataset


rfc_pred = Dclf.predict(tfidf_test)

score2=accuracy_score(y_test.astype('U').values,rfc_pred)
print(f'Decision Tree Classifier Accuracy: {round(score2*100,2)}%')
F1_score=f1_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Decision Trees F1-Score: {round(F1_score*100,2)}%')
Precision=precision_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Decision Trees Precision: {round(Precision*100,2)}%')
recall = recall_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Decision Trees Recall: {round(recall*100,2)}%')

print('')
 


from sklearn.neural_network import MLPClassifier
ANNModel=MLPClassifier(hidden_layer_sizes=(8, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=500, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)


#Train the model using the training sets
ANNModel.fit(tfidf_train,y_train.astype('U').values)

#Predict the response for test dataset

rfc_pred = ANNModel.predict(tfidf_test)

score2=accuracy_score(y_test.astype('U').values,rfc_pred)
print(f'Neural Network Accuracy: {round(score2*100,2)}%')
F1_score=f1_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Neural Network F1-Score: {round(F1_score*100,2)}%')
Precision=precision_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Neural Network Precision: {round(Precision*100,2)}%')
recall = recall_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Neural Network Recall: {round(recall*100,2)}%')

print('')


rf=RandomForestClassifier(n_estimators=10,n_jobs=2)
#Train the model using the training sets
rf.fit(tfidf_train,y_train.astype('U').values)

#Predict the response for test dataset

rfc_pred = rf.predict(tfidf_test)

score2=accuracy_score(y_test.astype('U').values,rfc_pred)
print(f'Random Forest Accuracy: {round(score2*100,2)}%')
F1_score=f1_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Random Forest  F1-Score: {round(F1_score*100,2)}%')
Precision=precision_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Random Forest  Precision: {round(Precision*100,2)}%')
recall = recall_score(y_test.astype('U').values, rfc_pred, average='weighted', zero_division=1)
print(f'Random Forest  Recall: {round(recall*100,2)}%')

print('')
#Ensemble
try:
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(tfidf_test, y_test.astype('U').values)
    print(f'Developed Model Accuracy: {round(result*100,2)}%')
except:    
    Ensemb = VotingClassifier( estimators = [('PAC',PAC),('SVM',SVMclf),('Decision Tree',Dclf),('ANN',ANNModel),('Random Forest',rf)], voting = 'hard')
    Ensemb.fit(tfidf_train,y_train.astype('U').values)
    Ensemb_pred=Ensemb.predict(tfidf_test)
    filename = 'finalized_model.sav'
    pickle.dump(Ensemb, open(filename, 'wb'))
    score2=accuracy_score(y_test.astype('U').values,Ensemb_pred)
    print(f'Ensemble Learning Accuracy: {round(score2*100,2)}%')
    F1_score=f1_score(y_test.astype('U').values, Ensemb_pred, average='weighted', zero_division=1)
    print(f'Ensemble Learning F1-Score: {round(F1_score*100,2)}%')
    Precision=precision_score(y_test.astype('U').values, Ensemb_pred, average='weighted', zero_division=1)
    print(f'Ensemble Learning Precision: {round(Precision*100,2)}%')
    recall = recall_score(y_test.astype('U').values, Ensemb_pred, average='weighted', zero_division=1)
    print(f'Ensemble Learning Recall: {round(recall*100,2)}%')


    


