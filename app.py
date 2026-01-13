
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="EmoDeteXt", page_icon="🎭", layout="centered")

# --- Constants ---
MODEL_FILE = 'emotion_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
METADATA_FILE = 'model_metadata.pkl'

# --- NLTK Setup ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')

download_nltk_data()

# --- Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# --- Data Loading (Only if training needed) ---
def load_and_prepare_data():
    try:
        df1 = pd.read_csv('text.csv')
        df2 = pd.read_csv('go_emotions_dataset.csv')
        
        label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        if 'label' in df1.columns:
            df1['emotion'] = df1['label'].map(label_map)
            df1 = df1[['text', 'emotion']]

        if 'example_very_unclear' in df2.columns:
            df2 = df2[df2['example_very_unclear'] == False]
        non_emotion_cols = ['id', 'text', 'example_very_unclear']
        emotion_cols = [c for c in df2.columns if c not in non_emotion_cols]
        df2['emotion'] = df2[emotion_cols].idxmax(axis=1)
        df2 = df2[['text', 'emotion']]

        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df.dropna(subset=['text', 'emotion'], inplace=True)
        df.drop_duplicates(subset=['text'], inplace=True)
        
        # Apply preprocessing
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        return df
    except FileNotFoundError:
        return None

# --- Model Management ---
@st.cache_resource
def get_model():
    # Check if model exists
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE) and os.path.exists(METADATA_FILE):
        model = joblib.load(MODEL_FILE)
        tfidf = joblib.load(VECTORIZER_FILE)
        metadata = joblib.load(METADATA_FILE)
        return model, tfidf, metadata['accuracy']
    
    # Train if not exists
    with st.spinner("Training model for the first time... This enables fast loads next time!"):
        df = load_and_prepare_data()
        if df is None:
            return None, None, None
            
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(df['cleaned_text'])
        y = df['emotion']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(max_iter=1000, multi_class='ovr')
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Save artifacts
        joblib.dump(model, MODEL_FILE)
        joblib.dump(tfidf, VECTORIZER_FILE)
        joblib.dump({'accuracy': acc}, METADATA_FILE)
        
        return model, tfidf, acc

# --- Main App ---
st.title("🎭 EmoDeteXt")
st.markdown("Enter a sentence below, and the AI will detect the emotional tone.")

model, tfidf, acc = get_model()

if model is None:
    st.error("Error: Dataset files not found. Ensure 'text.csv' and 'go_emotions_dataset.csv' are present.")
else:
    st.sidebar.success(f"Model Accuracy: {acc*100:.2f}%")
    st.sidebar.info("Model loaded from disk.")
    st.sidebar.markdown("### Detectable Emotions:")
    st.sidebar.write(", ".join(sorted(model.classes_)))

    user_input = st.text_area("How are you feeling?", placeholder="Type something here...")

    if st.button("Analyze Emotion"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            processed_input = preprocess_text(user_input)
            vec_input = tfidf.transform([processed_input])
            
            proba = model.predict_proba(vec_input)[0]
            prediction = model.predict(vec_input)[0]
            classes = model.classes_
            
            st.subheader(f"Prediction: **{prediction.upper()}**")
            
            if prediction == 'joy':
                st.balloons()
            
            prob_df = pd.DataFrame({'Emotion': classes, 'Probability': proba})
            st.bar_chart(prob_df.set_index('Emotion'))

st.markdown("---")
st.markdown("Built with Python & Streamlit")
