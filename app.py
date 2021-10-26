import streamlit as st
import pickle
import string
import base64
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier



tf_idf = pickle.load(open('tfvect.pkl','rb'))
model = pickle.load(open('model_PA.pkl','rb'))

# def fake_news_det(news):
#     input_data = [news]
#     vectorized_input_data = tfidf.transform(input_data)
#     prediction = model.predict(vectorized_input_data)
#     print(prediction)
    
# fake_news_det('Approximately 1:40 p.m. in circumstances that shotcrete was launched in the Nv. 1680 BP 255 of the OB2B, after finishing the launch of the first mixkret 113, the assistant of the alpha, Mr. Albertico asks the operator of the mixkret 113, Mr. Jhony to move the mixkret 116, so that access, finding in the cockpit of the mixkret the operator of the Launcher team, Mr. Danon asks him to come down. When the team started, he noticed that Mr. Danon (injured) was imprisoned between the team (height of the left rear rim) and the Hastial de la Labor.')
    
# print(fake_news_det)


# def run():
# title = st.title('Industrial Safety and health database')
st.markdown("<h1 style='text-align: center; color: black;'>Industrial Safety and health database</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Accident Level Predictor</h3>", unsafe_allow_html=True)
input_text = st.text_area("Enter Your Description Here 📰")



if st.button('Predict'):

    # 1. preprocess
    transformed_text= input_text
    # 2. vectorize
    vector_input = tf_idf.transform([transformed_text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 'I':
        st.header('Level-1')
    elif result == 'II':
        st.header("Level-2")
    elif result == 'III':
        st.header("Level-3")
    elif result == 'IV':
        st.header("Level-4")
    else:
        st.header("Level-5")