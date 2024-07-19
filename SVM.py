# SVM:- Support Vector Machine
import joblib
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import streamlit as st

# Define the preprocessing function
def preprocess_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return []
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    
    return tokens
#word tokenizer

def word_vectorizer(doc, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in doc:
        if word in model.wv:
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

svm=joblib.load("D:\\codes\\Python-Programs\\svm-model.pkl")
model=joblib.load("D:\\codes\\Python-Programs\\w2v.pkl")
test = st.text_input("Enter the string")
if test:
    t_v=preprocess_text(test)
    v=word_vectorizer(t_v, model, 100).reshape(1,-1)
    y = svm.predict(v)[0]
    print(y)
    st.write("-"*32)
    st.write(y)
else:
    st.write("")
