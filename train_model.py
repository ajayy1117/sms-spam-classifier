import pandas as pd
import numpy as np
import re
import string
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Set style for high quality visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Loading dataset...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Calculate metadata statistics
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Stratified Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_text'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )
    
    # TF-IDF Vectorization
    print("Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 1. Train Naive Bayes
    print("Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    nb_metrics = {
        'model': 'Multinomial Naive Bayes',
        'accuracy': float(accuracy_score(y_test, y_pred_nb)),
        'precision': float(precision_score(y_test, y_pred_nb)),
        'recall': float(recall_score(y_test, y_pred_nb)),
        'f1_score': float(f1_score(y_test, y_pred_nb))
    }
    
    # 2. Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    lr_metrics = {
        'model': 'Logistic Regression',
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr)),
        'recall': float(recall_score(y_test, y_pred_lr)),
        'f1_score': float(f1_score(y_test, y_pred_lr))
    }
    
    # Extract Feature Importances (Top words indicating spam vs ham)
    feature_names = np.array(vectorizer.get_feature_names_out())
    lr_coefs = lr_model.coef_[0]
    top_spam_indices = np.argsort(lr_coefs)[-20:][::-1]
    top_ham_indices = np.argsort(lr_coefs)[:20]
    
    top_spam_words = [{"word": feature_names[i], "score": float(lr_coefs[i])} for i in top_spam_indices]
    top_ham_words = [{"word": feature_names[i], "score": float(lr_coefs[i])} for i in top_ham_indices]
    
    # Save Confusion Matrices Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[0].set_title('Multinomial Naive Bayes Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[1].set_title('Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated confusion_matrices.png")
    
    # Save Dataset Insights & Feature Importance Charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Class Distribution
    sns.countplot(data=df, x='label', palette=['#10B981', '#EF4444'], ax=axes[0, 0])
    axes[0, 0].set_title('Class Distribution (Ham vs Spam)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Category')
    axes[0, 0].set_ylabel('Message Count')
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        
    # 2. Message Length Distribution
    sns.histplot(data=df, x='char_count', hue='label', bins=50, palette=['#10B981', '#EF4444'],
                 kde=True, ax=axes[0, 1], element='step')
    axes[0, 1].set_xlim(0, 300)
    axes[0, 1].set_title('Message Length Distribution (Characters)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Character Count')
    
    # 3. Top Spam Keywords
    spam_df = pd.DataFrame(top_spam_words[:12])
    sns.barplot(data=spam_df, x='score', y='word', palette='Reds_r', ax=axes[1, 0])
    axes[1, 0].set_title('Top 12 Words Indicating SPAM (Coefficients)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Logistic Regression Weight')
    axes[1, 0].set_ylabel('Keyword')
    
    # 4. Top Ham Keywords
    ham_df = pd.DataFrame(top_ham_words[:12])
    ham_df['abs_score'] = ham_df['score'].abs()
    sns.barplot(data=ham_df, x='abs_score', y='word', palette='Greens_r', ax=axes[1, 1])
    axes[1, 1].set_title('Top 12 Words Indicating HAM (Coefficients)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Magnitude of Weight towards Ham')
    axes[1, 1].set_ylabel('Keyword')
    
    plt.tight_layout()
    plt.savefig('dataset_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated dataset_insights.png")
    
    # Choose default best model
    best_model = nb_model if nb_metrics['f1_score'] >= lr_metrics['f1_score'] else lr_model
    best_name = "Multinomial Naive Bayes" if best_model == nb_model else "Logistic Regression"
    
    # Save both models & assets
    joblib.dump(best_model, 'spam_model.joblib')
    joblib.dump(nb_model, 'nb_model.joblib')
    joblib.dump(lr_model, 'lr_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    summary_data = {
        'naive_bayes': nb_metrics,
        'logistic_regression': lr_metrics,
        'best_model': best_name,
        'cm_nb': cm_nb.tolist(),
        'cm_lr': cm_lr.tolist(),
        'top_spam_words': top_spam_words,
        'top_ham_words': top_ham_words,
        'stats': {
            'total_samples': int(len(df)),
            'ham_count': int((df['target'] == 0).sum()),
            'spam_count': int((df['target'] == 1).sum()),
            'ham_avg_chars': float(df[df['target'] == 0]['char_count'].mean()),
            'spam_avg_chars': float(df[df['target'] == 1]['char_count'].mean()),
        }
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print("Training and asset generation complete!")

if __name__ == '__main__':
    main()
