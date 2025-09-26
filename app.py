import streamlit as st
from nltk.corpus import stopwords
from joblib import load
import nltk
import re
import base64

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    # it replaces everything except a-z and space with '' space
    text = re.sub(r'[^a-z\s]', '', text)
    # concatenates all words that are not in stop_words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

#xgb trained model from jupyter notebook
pipeline = load("model_trained.joblib")

image_path = "bg.jpg"
with open(image_path, "rb") as image:
    # read image(bianry format)
    data = image.read()
    # transform it into base64
    bytes = base64.b64encode(data)
    # transform it into string
    encoded_string = bytes.decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{encoded_string});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<span style='color: DarkRed; font-weight: bold; font-size: 40px;'> SMS Spam Detection</span>",
            unsafe_allow_html= True)
sms = st.text_input("", placeholder='Insert your SMS right here!')

if sms:
    text = clean_text(sms)
    prediction = pipeline.predict([text])
    if (prediction[0] == 0):
        st.image('ham.jpg')
    else:
        st.image("spam.jpg")