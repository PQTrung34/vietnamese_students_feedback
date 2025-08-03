import streamlit as st
import joblib
import os
from pyvi import ViTokenizer

# Load model
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, 'model')
model = joblib.load(os.path.join(model_dir, 'model.pkl'))
vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))

# Load stopwords
with open(os.path.join(os.path.join(current_dir, 'notebook'), 'vietnamese-stopwords.txt'), 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()
stopwords = ['_'.join(w.split()) for w in stopwords]

def preprocess(text):
    # Convert to lower
    text = text.lower()
    # Remove punctuation
    text = text.replace('[^\w\s]','')
    # Tokenization
    text = ViTokenizer.tokenize(text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# Streamlit app
st.set_page_config(
    page_title="Dự đoán cảm xúc phản hồi của học sinh",
)
st.title('Dự đoán cảm xúc phản hồi của học sinh')
# st.write("Dựa trên mô hình phân tích cảm xúc tiếng Việt")
text = st.text_area('Nhập phản hồi của học sinh:')
if st.button('Dự đoán'):
    text = preprocess(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    if prediction == 0:
        st.write('Phản hồi không tốt')
    elif prediction == 1:
        st.write('Phản hồi bình thường')
    else:
        st.write('Phản hồi tốt')