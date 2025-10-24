# app.py
import streamlit as st
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ---------------------------
# NLTK Setup
# ---------------------------
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root folder
MODELS_DIR = os.path.join(BASE_DIR, "models")

ovr_classifier_path = os.path.join(MODELS_DIR, "ovr_model.pkl")
tfidf_vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ---------------------------
# Check files exist
# ---------------------------
for path in [ovr_classifier_path, tfidf_vectorizer_path, label_encoder_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

# ---------------------------
# Load models
# ---------------------------
with open(ovr_classifier_path, "rb") as f:
    ovr_classifier = pickle.load(f)

with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# ---------------------------
# Text Preprocessing
# ---------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("📄 Resume Screening App")

st.write("Paste the resume text below and get a classification prediction.")

resume_text = st.text_area("Resume Text", height=300)

if st.button("Predict"):
    if not resume_text.strip():
        st.warning("Please enter a resume text!")
    else:
        # Preprocess
        processed_text = preprocess_text(resume_text)
        vectorized_text = tfidf_vectorizer.transform([processed_text])

        # Predict
        pred_class = ovr_classifier.predict(vectorized_text)
        pred_label = label_encoder.inverse_transform(pred_class)

        st.success(f"Predicted Category: **{pred_label[0]}**")
