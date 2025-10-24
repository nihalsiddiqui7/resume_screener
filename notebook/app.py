# app.py (Streamlit Resume Screener)
import streamlit as st
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords

# ---------------------------
# NLTK Setup
nltk.download("stopwords")
nltk.download("punkt")
STOPWORDS = set(stopwords.words("english"))

# ---------------------------
# Paths for models (robust for notebooks or Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # resume_screening_app/
MODELS_DIR = os.path.join(BASE_DIR, "models")

ovr_classifier_path = os.path.join(MODELS_DIR, "ovr_model.pkl")
tfidf_vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Check files exist
for path in [ovr_classifier_path, tfidf_vectorizer_path, label_encoder_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

# ---------------------------
# Load models
with open(ovr_classifier_path, "rb") as f:
    ovr_classifier = pickle.load(f)

with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# ---------------------------
# Helper functions
def preprocess_resume(text):
    """Clean and preprocess resume text"""
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def predict_resume_category(resume_text):
    """Predict the resume category"""
    processed_text = preprocess_resume(resume_text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    prediction = ovr_classifier.predict(vectorized_text)
    category = label_encoder.inverse_transform(prediction)[0]
    return category

# ---------------------------
# Streamlit App
st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="centered")
st.title("📄 Resume Screener App")
st.write("Paste the resume text below and get the predicted category!")

resume_input = st.text_area("Resume Text", height=300)

if st.button("Predict Category"):
    if resume_input.strip() == "":
        st.warning("Please enter some resume text to predict.")
    else:
        category = predict_resume_category(resume_input)
        st.success(f"Predicted Resume Category: **{category}**")

# Optional: sample resume for quick testing
if st.button("Load Sample Advocate Resume"):
    sample_advocate_resume = """
    Sarah Williams is a dedicated and skilled advocate with over 10 years of experience...
    """  # you can paste your full sample resume here
    st.text_area("Resume Text", value=sample_advocate_resume, height=300)
