# 🛡️ SMS Spam Detection Model & Web Application

An end-to-end Machine Learning project to classify SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing and Streamlit.

---

## 📌 Project Overview
This project fulfills the following requirements:
1. **Load and clean the SMS Spam Collection dataset**: Lowercase normalization, noise and punctuation removal, and tokenization.
2. **Feature Extraction**: Convert text to numerical features using **TF-IDF Vectorization** (`max_features=5000`, `ngram_range=(1, 2)`).
3. **Train and compare 2 models**: **Multinomial Naive Bayes** and **Logistic Regression**.
4. **Model Evaluation**: Evaluate both models using **Accuracy**, **Precision**, **Recall**, and a **Confusion Matrix**; select the better performing model.
5. **Interactive Web Application**: Build a **Streamlit** web app where a user can type any SMS message and get an instant spam/ham prediction.
6. **Limitations & Improvements**: Document model limitations (short messages, sarcasm, unseen slang) and explain how the system can be enhanced (Transformers, subword tokenization, feedback loops).

---

## 📊 Model Comparison & Results

Both models were trained on 80% of the dataset and evaluated on the remaining 20% (1,115 test samples):

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Selected)** | **96.68%** | **99.12%** | **75.84%** | **85.93%** |
| **Logistic Regression** | 96.68% | 100.00% | 75.17% | 85.82% |

> **Model Selection Decision:** **Multinomial Naive Bayes** was selected because it achieves higher **Recall** (75.84%) and **F1 Score** (85.93%), effectively catching more spam messages while maintaining high precision (99.12%).

---

## 📁 Repository Structure

```text
├── spam.csv                  # SMS Spam Collection Dataset
├── train_model.py            # Model training & evaluation script
├── spam_classifier.ipynb     # Interactive Jupyter Notebook
├── app.py                    # Streamlit Web Application
├── spam_model.joblib         # Serialized Trained Best Model
├── nb_model.joblib           # Serialized Naive Bayes Model
├── lr_model.joblib           # Serialized Logistic Regression Model
├── tfidf_vectorizer.joblib   # Serialized TF-IDF Vectorizer
├── model_metrics.json        # Exported evaluation metrics JSON
├── confusion_matrices.png    # Visual confusion matrices plot
├── assessment_report.md      # Consolidated Project Assessment Report
└── README.md                 # Project Documentation & Setup Guide
```

---

## 🚀 Quickstart & Setup Guide

### 1. Install Dependencies
Ensure Python 3.9+ is installed:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib streamlit pillow
```

### 2. Train Models (Optional)
To retrain the models and regenerate evaluation plots:
```bash
python train_model.py
```

### 3. Run the Streamlit Web Application
Launch the interactive web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚠️ Model Limitations & Future Improvements

### Limitations:
- **Short Messages**: Very short texts (*"ok"*, *"yes"*, *"call"*) have sparse TF-IDF features.
- **Sarcasm & Context**: Word-level representations do not capture contextual irony.
- **Unseen Slang / Typos**: Obfuscated words (`c@sh`, `w1nner`, `fr33`) create out-of-vocabulary tokens.

### Future Improvements:
- Fine-tune pre-trained **Transformer models** (BERT / DistilBERT / RoBERTa) for contextual embeddings.
- Implement **Subword Tokenization** (BPE) to handle intentional character obfuscation.
- Add an active **User Feedback Loop** for continuous model retraining.
