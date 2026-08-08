import os
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)

def create_report_pdf(filename="SMS_Spam_Detection_Assessment_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )
    story = []
    
    # Color palette
    PRIMARY_RED = colors.HexColor("#DC2626")
    DARK_TEXT = colors.HexColor("#0F172A")
    BODY_TEXT = colors.HexColor("#334155")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E1")

    # Typography / Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY_RED,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=BODY_TEXT,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY_RED,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=BODY_TEXT,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=BODY_TEXT,
        leftIndent=12,
        spaceAfter=4
    )

    link_style = ParagraphStyle(
        'Link_Custom',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1D4ED8")
    )

    # -------------------------------------------------------------
    # 1. Header & Title Block
    # -------------------------------------------------------------
    story.append(Paragraph("SMS Spam Detection System — Assessment Report", title_style))
    story.append(Paragraph("<b>Domain:</b> Artificial Intelligence & Machine Learning (AI/ML) | <b>Task:</b> SMS Spam Classification & Streamlit App", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_RED, spaceAfter=10))

    # -------------------------------------------------------------
    # 2. GitHub Repository Link Box
    # -------------------------------------------------------------
    repo_data = [[
        Paragraph("<b>GitHub Repository:</b> <a href='https://github.com/ajayy1117/sms-spam-classifier'><u>https://github.com/ajayy1117/sms-spam-classifier</u></a>", link_style)
    ]]
    repo_table = Table(repo_data, colWidths=[532])
    repo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEF2F2")),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY_RED),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(repo_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 3. Section 1: Project Description & Approach
    # -------------------------------------------------------------
    story.append(Paragraph("1. Project Description & Engineering Approach", h1_style))
    story.append(Paragraph(
        "This project implements an end-to-end Machine Learning pipeline to classify SMS text messages as <b>Spam</b> or <b>Ham (Legitimate)</b> using Natural Language Processing (NLP) techniques, bundled in an interactive desktop Streamlit web application.",
        body_style
    ))
    story.append(Paragraph("<b>Key Pipeline Stages:</b>", body_style))
    story.append(Paragraph("• <b>Data Ingestion & Cleaning:</b> Loaded the SMS Spam Collection dataset (5,572 messages). Standardized text with lowercasing, regex noise/URL removal, punctuation stripping, and tokenization.", bullet_style))
    story.append(Paragraph("• <b>Feature Extraction:</b> Applied <b>TF-IDF Vectorization</b> (max_features=5000, unigrams and bigrams, English stopword filtering) to convert textual data into high-dimensional numerical matrices.", bullet_style))
    story.append(Paragraph("• <b>Model Training:</b> Trained and compared two core algorithms: <b>Multinomial Naive Bayes</b> and <b>Logistic Regression</b> using a stratified 80/20 train/test split.", bullet_style))
    story.append(Paragraph("• <b>Web Application:</b> Developed a responsive Streamlit application featuring real-time message analysis, confidence progress indicators, and explainable keyword attribution.", bullet_style))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 4. Section 2: Model Evaluation & Key Results
    # -------------------------------------------------------------
    story.append(Paragraph("2. Model Evaluation & Performance Comparison", h1_style))
    
    # Comparison Table
    table_content = [
        [Paragraph("<b>Evaluation Metric</b>", body_style), Paragraph("<b>Multinomial Naive Bayes (Selected)</b>", body_style), Paragraph("<b>Logistic Regression</b>", body_style)],
        [Paragraph("Accuracy", body_style), Paragraph("<b>96.68%</b>", body_style), Paragraph("96.68%", body_style)],
        [Paragraph("Precision", body_style), Paragraph("99.12%", body_style), Paragraph("<b>100.00%</b>", body_style)],
        [Paragraph("Recall", body_style), Paragraph("<b>75.84%</b>", body_style), Paragraph("75.17%", body_style)],
        [Paragraph("F1 Score", body_style), Paragraph("<b>85.93%</b>", body_style), Paragraph("85.82%", body_style)]
    ]
    perf_table = Table(table_content, colWidths=[172, 180, 180])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER')
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Model Selection Decision:</b> <b>Multinomial Naive Bayes</b> was selected as the superior production model due to its higher <b>Recall (75.84%)</b> and <b>F1 Score (85.93%)</b>, successfully capturing more spam messages while maintaining an exceptionally high precision of 99.12%.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # 5. Section 3: Visual Charts & Outputs
    # -------------------------------------------------------------
    story.append(Paragraph("3. Visual Output & Confusion Matrices", h1_style))
    if os.path.exists("confusion_matrices.png"):
        story.append(RLImage("confusion_matrices.png", width=480, height=190))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 6. Section 4: Key Insights & Recommendations
    # -------------------------------------------------------------
    story.append(Paragraph("4. Key Insights, Challenges & Recommendations", h1_style))
    story.append(Paragraph("<b>Identified Limitations:</b>", body_style))
    story.append(Paragraph("1. <b>Short Message Sparsity:</b> Texts under 5 words (e.g. <i>'ok'</i>, <i>'call me'</i>) lack sufficient TF-IDF feature density, leading to lower classification confidence.", bullet_style))
    story.append(Paragraph("2. <b>Sarcasm & Word Order:</b> Bag-of-Words ignores sequence structure and syntax, making contextual sarcasm challenging to detect without sentence-level embeddings.", bullet_style))
    story.append(Paragraph("3. <b>Unseen Slang & Typos:</b> Spammers frequently use character substitutions (<code>c@sh</code>, <code>w1nner</code>) that generate out-of-vocabulary (OOV) tokens in standard word-level models.", bullet_style))
    
    story.append(Paragraph("<b>Suggestions for Improvement:</b>", body_style))
    story.append(Paragraph("• <b>Transformer Fine-Tuning:</b> Deploy pre-trained contextual models like <b>BERT</b> or <b>RoBERTa</b> to capture complex sentence semantics.", bullet_style))
    story.append(Paragraph("• <b>Subword Tokenization:</b> Use Byte-Pair Encoding (BPE) or character n-grams to mitigate typo and character-obfuscation evasion.", bullet_style))
    story.append(Paragraph("• <b>Active Learning Loop:</b> Integrate user feedback reporting in the web app to log misclassified instances for continuous automated model retraining.", bullet_style))

    # Build Document
    doc.build(story)
    print(f"PDF successfully generated: {filename}")

if __name__ == '__main__':
    create_report_pdf()
