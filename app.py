import streamlit as st
import joblib
import json
import re
import string
import os
import pandas as pd
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Application Setup & Desktop-Optimized Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Clean Red & White Desktop Styling (Zero HTML Tag Collisions)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit Header, Toolbar, Menu, and Share Icons */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}

    /* Desktop Layout Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1300px !important;
    }

    /* Top Header Bar */
    .brand-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 1.2rem;
        border-bottom: 1.5px solid #E2E8F0;
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .brand-title span {
        color: #DC2626;
    }
    .brand-sub {
        font-size: 0.9rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Result Badges */
    .spam-badge {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .spam-badge-title {
        color: #DC2626;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .spam-badge-desc {
        color: #991B1B;
        font-size: 0.875rem;
        margin: 0;
    }

    .ham-badge {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .ham-badge-title {
        color: #16A34A;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .ham-badge-desc {
        color: #166534;
        font-size: 0.875rem;
        margin: 0;
    }

    .idle-badge {
        background-color: #F8FAFC;
        border: 1.5px dashed #CBD5E1;
        border-radius: 10px;
        padding: 3rem 1.5rem;
        text-align: center;
        color: #64748B;
        font-size: 0.95rem;
    }

    /* Keyword Tags */
    .tag-spam {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        font-size: 0.825rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #FECACA;
        margin: 3px;
    }
    .tag-ham {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        font-size: 0.825rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #BBF7D0;
        margin: 3px;
    }

    /* Primary Red Button */
    .stButton>button[kind="primary"] {
        background-color: #DC2626 !important;
        border-color: #DC2626 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.25rem !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #B91C1C !important;
        border-color: #B91C1C !important;
    }

    /* Secondary Buttons */
    .stButton>button {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        font-weight: 600 !important;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1.5px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #64748B !important;
    }
    .stTabs [aria-selected="true"] {
        color: #DC2626 !important;
        font-weight: 700 !important;
        border-bottom: 2.5px solid #DC2626 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Preprocessing & Model Loading Utilities
# -----------------------------------------------------------------------------
def clean_input_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return " ".join(tokens)

@st.cache_resource
def load_assets():
    nb_model = joblib.load('nb_model.joblib') if os.path.exists('nb_model.joblib') else joblib.load('spam_model.joblib')
    lr_model = joblib.load('lr_model.joblib') if os.path.exists('lr_model.joblib') else None
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    metrics = None
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
            
    return nb_model, lr_model, vectorizer, metrics

try:
    nb_model, lr_model, vectorizer, metrics = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading model dependencies: {e}. Please run `python train_model.py` first.")

# -----------------------------------------------------------------------------
# Top Header Bar (With Proper Desktop Spacing)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-title">🛡️ SMS <span>Spam Classifier</span></div>
        <div class="brand-sub">Machine Learning & Natural Language Processing Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_compare, tab_limitations = st.tabs([
    "💬 Message Classifier",
    "📊 Model Evaluation & Benchmarks",
    "⚠️ Limitations & Technical Roadmap"
])

# =============================================================================
# TAB 1: MESSAGE CLASSIFIER (Desktop Split Grid)
# =============================================================================
with tab_predict:
    col_input, col_result = st.columns([1.2, 0.95], gap="large")

    # ------------------ LEFT COLUMN: Message Input ------------------
    with col_input:
        with st.container(border=True):
            st.subheader("Input Message")
            
            # Model Dropdown
            selected_model = st.selectbox(
                "Classification Algorithm:",
                ["Multinomial Naive Bayes", "Logistic Regression"],
                index=0
            )

            # Preset Test Buttons
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #475569; margin: 10px 0 4px 0;'>Quick Test Presets:</p>", unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            if p1.button("🎰 Prize Scam", use_container_width=True):
                st.session_state['sms_input'] = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim."
            if p2.button("💳 Bank Alert", use_container_width=True):
                st.session_state['sms_input'] = "URGENT! Your Mobile number has won £2000 Bonus Cash. Call 09050000321 now to claim your code. Valid 12hrs only."
            if p3.button("☕ Legitimate (Ham)", use_container_width=True):
                st.session_state['sms_input'] = "Hey mate, are we still meeting up for coffee around 4 pm today?"

            # Text Area Input
            default_val = st.session_state.get('sms_input', '')
            user_message = st.text_area(
                "SMS Text Area:",
                value=default_val,
                height=130,
                placeholder="Type or paste the SMS text message here...",
                label_visibility="collapsed"
            )

            # Action Buttons
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                run_classify = st.button("🚀 Classify Message", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state['sms_input'] = ""
                    st.rerun()

    # ------------------ RIGHT COLUMN: Live Diagnostics ------------------
    with col_result:
        with st.container(border=True):
            st.subheader("Classification Result")

            if (run_classify or user_message) and user_message.strip() and assets_loaded:
                cleaned_msg = clean_input_text(user_message)
                features = vectorizer.transform([cleaned_msg])
                
                active_model = nb_model if selected_model == "Multinomial Naive Bayes" else (lr_model if lr_model else nb_model)
                prediction = active_model.predict(features)[0]
                probabilities = active_model.predict_proba(features)[0] if hasattr(active_model, 'predict_proba') else [0.5, 0.5]

                # Verdict Display
                if prediction == 1:
                    st.markdown("""
                    <div class="spam-badge">
                        <div style="font-size: 1.8rem; margin-bottom: 2px;">🚨</div>
                        <div class="spam-badge-title">SPAM DETECTED</div>
                        <p class="spam-badge-desc">This message matches patterns typical of spam, promotional offers, or scams.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="ham-badge">
                        <div style="font-size: 1.8rem; margin-bottom: 2px;">✅</div>
                        <div class="ham-badge-title">LEGITIMATE (HAM)</div>
                        <p class="ham-badge-desc">This message appears authentic and does not trigger spam filters.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence Gauge
                st.markdown("<p style='font-size: 0.875rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>Confidence Scores:</p>", unsafe_allow_html=True)
                st.write(f"Spam Probability: **{probabilities[1]*100:.1f}%**")
                st.progress(probabilities[1])
                st.write(f"Ham Probability: **{probabilities[0]*100:.1f}%**")
                st.progress(probabilities[0])

                # Keyword Diagnostics
                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.875rem; font-weight: 600; color: #334155; margin-bottom: 6px;'>Key Indicator Words:</p>", unsafe_allow_html=True)
                
                words_in_msg = cleaned_msg.split()
                spam_words_detected = []
                ham_words_detected = []

                if metrics:
                    top_spam_dict = {item['word']: item['score'] for item in metrics.get('top_spam_words', [])}
                    top_ham_dict = {item['word']: item['score'] for item in metrics.get('top_ham_words', [])}

                    for w in set(words_in_msg):
                        if w in top_spam_dict:
                            spam_words_detected.append(w)
                        elif w in top_ham_dict:
                            ham_words_detected.append(w)

                if spam_words_detected:
                    tags_html = "".join([f'<span class="tag-spam">⚠️ {w}</span>' for w in spam_words_detected])
                    st.markdown(tags_html, unsafe_allow_html=True)
                elif ham_words_detected:
                    tags_html = "".join([f'<span class="tag-ham">✓ {w}</span>' for w in ham_words_detected])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size: 0.85rem; color: #64748B;'>No strong keyword indicators detected.</p>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="idle-badge">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">✉️</div>
                    <div style="font-weight: 600; color: #334155; margin-bottom: 4px;">Ready to Classify</div>
                    Type an SMS message on the left or select a preset prompt, then click <b>Classify Message</b>.
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: MODEL EVALUATION & BENCHMARKS
# =============================================================================
with tab_compare:
    if metrics:
        nb_m = metrics['naive_bayes']
        lr_m = metrics['logistic_regression']

        # 4 Desktop Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.caption("SELECTED MODEL")
                st.subheader(metrics['best_model'])
        with m2:
            with st.container(border=True):
                st.caption("NAIVE BAYES ACCURACY")
                st.subheader(f"{nb_m['accuracy']*100:.2f}%")
        with m3:
            with st.container(border=True):
                st.caption("LOGISTIC REG ACCURACY")
                st.subheader(f"{lr_m['accuracy']*100:.2f}%")
        with m4:
            with st.container(border=True):
                st.caption("NAIVE BAYES RECALL")
                st.subheader(f"{nb_m['recall']*100:.2f}%")

        # Desktop 2-Column Split: Table & Confusion Matrix
        col_table, col_cm = st.columns([1.1, 0.9], gap="large")
        
        with col_table:
            with st.container(border=True):
                st.subheader("Performance Comparison (1,115 Test Samples)")
                
                bench_df = pd.DataFrame({
                    "Evaluation Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Multinomial Naive Bayes": [
                        f"{nb_m['accuracy']*100:.2f}%",
                        f"{nb_m['precision']*100:.2f}%",
                        f"{nb_m['recall']*100:.2f}%",
                        f"{nb_m['f1_score']:.4f}"
                    ],
                    "Logistic Regression": [
                        f"{lr_m['accuracy']*100:.2f}%",
                        f"{lr_m['precision']*100:.2f}%",
                        f"{lr_m['recall']*100:.2f}%",
                        f"{lr_m['f1_score']:.4f}"
                    ]
                })
                st.table(bench_df)
                
                st.info(f"💡 **Model Selection Rationale:** **{metrics['best_model']}** was chosen as the primary classifier because of its higher Recall ({nb_m['recall']*100:.2f}%) and F1-Score ({nb_m['f1_score']:.4f}), which ensures fewer spam messages bypass detection.")

        with col_cm:
            with st.container(border=True):
                st.subheader("Confusion Matrices")
                if os.path.exists("confusion_matrices.png"):
                    cm_img = Image.open("confusion_matrices.png")
                    st.image(cm_img, use_container_width=True)

# =============================================================================
# TAB 3: LIMITATIONS & TECHNICAL ROADMAP
# =============================================================================
with tab_limitations:
    col_lim, col_road = st.columns([1, 1], gap="large")

    with col_lim:
        with st.container(border=True):
            st.subheader("⚠️ Model Limitations")
            st.markdown("""
            1. **Short Message Sparsity**:
               - Very short messages (e.g. *"ok"*, *"call me"*, *"yes"*) contain sparse TF-IDF vectors, reducing predictive certainty.
            2. **Sarcasm & Semantic Ordering**:
               - TF-IDF ignores word order and sentence syntax. Sarcastic spam containing positive words may be misclassified.
            3. **Unseen Slang & Typos**:
               - Intentional character substitutions (`c@sh`, `w1nner`, `fr33`) create out-of-vocabulary (OOV) tokens that standard tokenizers cannot resolve.
            4. **Class Imbalance**:
               - The dataset consists of ~87% Ham and ~13% Spam, requiring precision, recall, and F1-score to be monitored closely.
            """)

    with col_road:
        with st.container(border=True):
            st.subheader("🚀 Technical Roadmap")
            st.markdown("""
            1. **Pre-trained Transformer Encoders**:
               - Fine-tune contextual models such as **BERT**, **DistilBERT**, or **RoBERTa** to capture bidirectional context and subtle phishing intent.
            2. **Subword & Character Tokenization**:
               - Implement Byte-Pair Encoding (BPE) or character n-grams to identify character substitutions and obfuscated keywords.
            3. **Active User Feedback Loop**:
               - Add an in-app reporting mechanism where misclassified messages can be submitted to continuously retrain the model.
            """)
