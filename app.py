import streamlit as st
from nltk.corpus import stopwords
import nltk
from joblib import load

nltk.download('stopwords')
import re

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # elimină tot ce nu e literă sau spațiu
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


pipeline = load("model_trained.joblib")

st.write("Introduceti textul")
sms = st.text_input("SMS Detection")

if sms:
    text = clean_text(sms)
    st.write("Text curățat:", text)
    prediction = pipeline.predict([text])
    st.write("Predicție:", prediction[0])
