
# EmoDeteXt: AI-Powered Emotion Detection from Text 🎭

A machine learning project that detects emotions (Joy, Sadness, Anger, Fear, Love, Surprise) from textual data using Natural Language Processing (NLP).

## 📌 Project Overview
This project builds an AI model capable of classifying the emotional tone of text. It combines multiple datasets, applies rigorous text preprocessing, and utilizes TF-IDF vectorization with Logistic Regression for accurate multi-class classification.

## 📂 Repository Structure
*   `emotion_detection_project.ipynb`: The main Jupyter Notebook with the end-to-end workflow (Data analysis, training, evaluation).
*   `project_pipeline.py`: A Python script version of the project for command-line execution.
*   `REPORT.md`: A detailed report explaining the methodology, model choices, and evaluation metrics.
*   `requirements.txt`: List of dependencies required to run the project.

## ⚡ Features
*   **Data Integration**: Merges dataset inputs (integer-labeled and one-hot encoded) into a unified format.
*   **NLP Pipeline**: Lowercasing, noise removal, tokenization, stopword removal, and lemmatization.
*   **Modeling**: Logistic Regression with TF-IDF features.
*   **Real-time Prediction**: Includes a function for testing custom user inputs.

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
Run the pipeline script:
```bash
python project_pipeline.py
```
Or open `emotion_detection_project.ipynb` in Jupyter Notebook/Lab to explore the step-by-step implementation.

## 📊 Results
The model evaluates text based on 6 core emotions using Accuracy and F1-Score metrics. Visualizations include confusion matrices to analyze misclassifications.

---

