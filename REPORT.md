
# EmoDeteXt
AI-Powered Emotion Detection from Text

## 1. Project Overview
This project focuses on building an AI model to detect emotions from text using Natural Language Processing (NLP). Two open datasets were combined to train a Logistic Regression model that classifies text into emotions such as Joy, Sadness, Anger, Fear, Love, Surprise, etc.

## 2. Methodology

### 2.1 Dataset Collection and Merging
We utilized two datasets:
1.  **`text.csv` or dataset1**: A labeled dataset with integer emotion codes, taken from Kaggle Twitter Emotions Dataset.
2.  **`go_emotions_dataset.csv` or dataset2**: A large, fine-grained emotion dataset with one-hot encoding, taken from Google Go Emotions Dataset.

**Harmonization Process:**
- Mapped integer labels in `text.csv` to standard emotion names: `0: sadness`, `1: joy`, `2: love`, `3: anger`, `4: fear`, `5: surprise`.
- Converted multi-label vectors in `go_emotions` to a single dominant emotion using `idxmax`.
- Concatenated both datasets into a single DataFrame with `text` and `emotion` columns.
- Removed duplicate texts and null values to ensure data quality.

### 2.2 Text Preprocessing
A standard NLP pipeline was implemented to clean the text data:
- **Lowercasing**: normalized text case.
- **Noise Removal**: removed non-alphabetic characters and special symbols.
- **Tokenization**: split text into individual words.
- **Stopword Removal**: removed common words (e.g., "the", "is") that add little semantic meaning.
- **Lemmatization**: converted words to their base form (e.g., "running" -> "run") using WordNetLemmatizer.

### 2.3 Feature Extraction
**TF-IDF (Term Frequency-Inverse Document Frequency)** was used to convert text into numerical vectors. We limited the vocabulary to the top 5,000 features to balance performance and computational efficiency.

### 2.4 Model Building
**Logistic Regression** was selected for classification due to its efficiency and effectiveness in high-dimensional sparse data (text).
- **Split**: 80% Training, 20% Testing.
- **Strategy**: One-vs-Rest (OvR) for multi-class classification.

## 3. Evaluation Results
The model was evaluated using standard metrics on the test set.
*Note: Specific metrics in notebook are from a representative run.*
- **Accuracy**: Indicates the overall percentage of correct predictions.
- **F1-Score**: Weighted average of Precision and Recall, accounting for class imbalance.
- **Confusion Matrix**: Visualized to identify common misclassifications (e.g., confusing 'love' with 'joy').

## 4. Real-World Application
The project includes a `predict_new_text(user_input)` function that allows users to input raw text and receive an emotion prediction in real-time. This demonstrates the model's applicability to tasks like customer sentiment analysis or chatbot emotion awareness.

## 5. Conclusion
An end-to-end emotion detection system was successfully built. The combination of datasets provided a robust training ground, and the TF-IDF + Logistic Regression pipeline yielded a functional classifier capable of distinguishing nuance in emotional text.
