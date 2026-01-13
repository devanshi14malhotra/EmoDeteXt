
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
import time

# --- Page Config ---
st.set_page_config(page_title="EmoDeteXt", page_icon="🎭", layout="centered")

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
        nltk.download('omw-1.4') # Often needed for lemmatizer

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

# --- Data Loading & Merging (Cached) ---
@st.cache_data
def load_and_prepare_data():
    try:
        # Load datasets
        df1 = pd.read_csv('text.csv')
        df2 = pd.read_csv('go_emotions_dataset.csv')
        
        # Process df1
        label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        if 'label' in df1.columns:
            df1['emotion'] = df1['label'].map(label_map)
            df1 = df1[['text', 'emotion']]

        # Process df2
        if 'example_very_unclear' in df2.columns:
            df2 = df2[df2['example_very_unclear'] == False]
        non_emotion_cols = ['id', 'text', 'example_very_unclear']
        emotion_cols = [c for c in df2.columns if c not in non_emotion_cols]
        df2['emotion'] = df2[emotion_cols].idxmax(axis=1)
        df2 = df2[['text', 'emotion']]

        # Merge
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.dropna(subset=['text', 'emotion'], inplace=True)
        df.drop_duplicates(subset=['text'], inplace=True)
        
        # Apply preprocessing
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        return df
    except FileNotFoundError:
        return None

# --- Model Training (Cached) ---
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['emotion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, tfidf, acc

# --- Main App Interface ---
st.title("🎭 EmoDeteXt")
st.markdown("Enter a sentence below, and the AI will detect the underlying emotion.")

# Load Data
with st.spinner("Loading and preparing datasets... this might take a moment on first run."):
    df = load_and_prepare_data()

if df is None:
    st.error("Error: Dataset files ('text.csv', 'go_emotions_dataset.csv') not found in the directory.")
else:
    # Train Model
    with st.spinner("Training the AI model..."):
        model, tfidf, acc = train_model(df)
    
    st.sidebar.success(f"Model trained with {acc*100:.2f}% accuracy!")
    st.sidebar.markdown("### Emotions it can detect:")
    st.sidebar.write(", ".join(sorted(df['emotion'].unique())))

    # User Input
    user_input = st.text_area("How are you feeling?", placeholder="Type something here... e.g., 'I got the job and I am so happy!'")

    if st.button("Analyze Emotion"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Prediction
            processed_input = preprocess_text(user_input)
            vec_input = tfidf.transform([processed_input])
            
            # Get probabilities
            proba = model.predict_proba(vec_input)[0]
            prediction = model.predict(vec_input)[0]
            classes = model.classes_
            
            # Display Result
            st.subheader(f"Prediction: **{prediction.upper()}**")
            
            if prediction == 'joy':
                st.balloons()
            
            # Probability Chart
            prob_df = pd.DataFrame({'Emotion': classes, 'Probability': proba})
            st.bar_chart(prob_df.set_index('Emotion'))

st.markdown("---")
st.markdown("Built with Python & Streamlit")
