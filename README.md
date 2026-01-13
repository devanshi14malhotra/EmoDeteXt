# EmoDeteXt: AI-Powered Emotion Detection from Text 🎭
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://emodetext-devanshimalhotra.streamlit.app/)
![Status](https://img.shields.io/badge/Status-Live-success)

A machine learning project that detects emotions (Joy, Sadness, Anger, Fear, Love, Surprise) from textual data using Natural Language Processing (NLP).

## 📌 Project Overview
This project builds an AI model capable of classifying the emotional tone of text. It combines multiple datasets, applies rigorous text preprocessing, and utilizes TF-IDF vectorization with Logistic Regression for accurate multi-class classification.

## 📂 Repository Structure
*   `emotion_detection_project.ipynb`: The main Jupyter Notebook with the end-to-end workflow (Data analysis, training, evaluation).
*   `app.py`: The main Streamlit application file used to run and deploy the EmoDeteXt web app.
*   `REPORT.md`: A detailed report explaining the methodology, model choices, and evaluation metrics.
*   `requirements.txt`: List of dependencies required to run the project.
*   `*.pkl`: Pre-trained machine learning models and vectorizers used for emotion prediction.

## ⚡ Features
*   **Web Deployment**: Fully interactive web application built with Streamlit and deployed to the cloud.
*   **Real-time Prediction**: Type any sentence and get instant emotion analysis with probability confidence scores.
*   **Data Integration**: Merges dataset inputs (integer-labeled and one-hot encoded) into a unified format.
*   **NLP Pipeline**: Lowercasing, noise removal, tokenization, stopword removal, and lemmatization.
*   **Modeling**: Logistic Regression with TF-IDF features.

## 🚀 Getting Started

### Prerequisites
*   Python 3.x
*   PIP

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/devanshi14malhotra/EmoDeteXt.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

##### **Option 1: Run Locally**

Run the Streamlit app on your local machine:
```bash
streamlit run app.py
```

##### **Option 2: Access Online**

Experience the live application without installation:
👉 **[EmoDeteXt Web App](https://emodetext-devanshimalhotra.streamlit.app/)**

##### **Option 3: Explore the Code**

Open `emotion_detection_project.ipynb` in Jupyter Notebook/Lab to explore the data analysis and training steps.

## 📊 Results
The model evaluates text based on the core emotions using Accuracy and F1-Score metrics. Visualizations include confusion matrices to analyze misclassifications.

---

