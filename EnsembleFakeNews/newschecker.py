import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
from wordcloud import WordCloud
# import the streamlit library
import streamlit as st
 
# title of the  app
st.header('Welcome to Ensemble-Based Misinformation Detection System')

newstitle = st.text_input("Enter the news title ", "Paste Title Here ...")

newstext = st.text_input("Copy and Paste the news content here ",key= "content")
newscheck=st.button('Predict')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=1)
if(newscheck):
     st.text("Checking the Genuity of {}.".format(newstitle))
     
     filename = 'finalized_model.sav'
     loaded_model = pickle.load(open(filename, 'rb'))
     newschecking =  newstitle + ' ' + newstext
     newschecking=[newschecking]
     tfidf_train = tfidf_vectorizer.fit_transform(newschecking)
     
     textab=tfidf_vectorizer.transform(newschecking)
     pridi=loaded_model.predict(textab)
     pre='pridi'
     if(pre =='Fake'):
         st.warning("The News is False, Kindly Ignore!")
     elif(pre == 'REAL'):
        st.success("The News is Genuine")