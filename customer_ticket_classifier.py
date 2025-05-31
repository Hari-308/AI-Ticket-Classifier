# Necessary imports
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from textblob import TextBlob
from dateutil import parser as date_parser
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

# Download NLTK only if not already downloaded
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Load data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    df = df.dropna(subset=['ticket_id'])
    df['urgency_level'] = df['urgency_level'].fillna('High')
    df = df.dropna(subset=['issue_type', 'product'], how='all')
    df['ticket_text'] = df['ticket_text'].fillna('')
    return df

df = load_and_clean_data('ai_dev_assignment_tickets_complex_1000.xlsx.csv')

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['ticket_text'].apply(preprocess_text)

# Feature engineering
def extract_features(df):
    df['text_length'] = df['ticket_text'].str.len()
    df['word_count'] = df['ticket_text'].str.split().str.len()
    df['contains_urgent'] = df['ticket_text'].str.contains('urgent|emergency|immediately|asap', case=False, na=False).astype(int)
    df['contains_question'] = df['ticket_text'].str.contains('\?', na=False).astype(int)
    df['sentiment'] = df['ticket_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['product'] = df['product'].astype('category')
    return df

df = extract_features(df)

# Entity extractor (replaces datefinder)
def extract_dates(text):
    date_regex = r'(\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\b\d{4}-\d{2}-\d{2}\b))'
    matches = re.findall(date_regex, text, flags=re.IGNORECASE)
    parsed_dates = []
    for date_str in matches:
        try:
            parsed_dates.append(str(date_parser.parse(date_str, fuzzy=True)))
        except:
            continue
    return parsed_dates

def extract_entities(text):
    if not isinstance(text, str):
        return {"products": None, "dates": [], "complaint_keywords": []}
    products = ['SmartWatch V2', 'UltraClean Vacuum', 'SoundWave 300', 
                'PhotoSnap Cam', 'Vision LED TV', 'RoboChef Blender', 
                'FitRun Treadmill', 'PowerMax Battery', 'EcoBreeze AC', 'ProTab X1']
    complaint_keywords = ['broken', 'damaged', 'not working', 'error', 'issue', 
                          'problem', 'late', 'missing', 'wrong', 'defect']
    found_products = [p for p in products if p.lower() in text.lower()]
    dates = extract_dates(text)
    complaints = [kw for kw in complaint_keywords if kw in text.lower()]
    return {
        'products': found_products[0] if found_products else None,
        'dates': dates,
        'complaint_keywords': complaints
    }

# ML preprocessing and model
X = df[['clean_text', 'text_length', 'word_count', 'contains_urgent', 
        'contains_question', 'sentiment', 'product']]
y_issue = df['issue_type']
y_urgency = df['urgency_level']
le = LabelEncoder()
y_urgency_encoded = le.fit_transform(y_urgency)

X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
    X, y_issue, y_urgency_encoded, test_size=0.2, random_state=42, stratify=y_urgency_encoded
)

text_features = 'clean_text'
numeric_features = ['text_length', 'word_count', 'contains_urgent', 'contains_question', 'sentiment']
categorical_features = ['product']

preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), text_features),
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

issue_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42))
])

urgency_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42))
])

issue_pipeline.fit(X_train, y_issue_train)
urgency_pipeline.fit(X_train, y_urgency_train)

# Evaluation
def evaluate_models():
    y_issue_pred = issue_pipeline.predict(X_test)
    y_urgency_pred = urgency_pipeline.predict(X_test)
    print("Issue Type Classification Report:")
    print(classification_report(y_issue_test, y_issue_pred))
    print("\nUrgency Level Classification Report:")
    print(classification_report(y_urgency_test, y_urgency_pred, target_names=le.classes_))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(confusion_matrix(y_issue_test, y_issue_pred), annot=True, fmt='d', ax=ax1)
    ax1.set_title('Issue Type Confusion Matrix')
    sns.heatmap(confusion_matrix(y_urgency_test, y_urgency_pred), annot=True, fmt='d', ax=ax2)
    ax2.set_title('Urgency Level Confusion Matrix')
    plt.show()

evaluate_models()

# Prediction
def predict_ticket(ticket_text):
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        return {"error": "Invalid ticket text"}
    
    clean_text = preprocess_text(ticket_text)
    entities = extract_entities(ticket_text)
    product = entities['products'] if entities['products'] else 'SmartWatch V2'  # Default to avoid OneHotEncoder crash

    features = pd.DataFrame({
        'clean_text': [clean_text],
        'text_length': [len(ticket_text)],
        'word_count': [len(ticket_text.split())],
        'contains_urgent': [int(bool(re.search('urgent|emergency|immediately|asap', ticket_text, re.I)))],
        'contains_question': [int('?' in ticket_text)],
        'sentiment': [TextBlob(ticket_text).sentiment.polarity],
        'product': [product]
    })

    issue_pred = issue_pipeline.predict(features)[0]
    urgency_pred = le.inverse_transform(urgency_pipeline.predict(features))[0]
    urgency_proba = max(urgency_pipeline.predict_proba(features)[0])
    
    return {
        "ticket_text": ticket_text,
        "predicted_issue": issue_pred,
        "predicted_urgency": urgency_pred,
        "urgency_confidence": float(urgency_proba),
        "extracted_entities": entities
    }

# Gradio interface
def gradio_predict(ticket_text):
    result = predict_ticket(ticket_text)
    return {
        "Issue Type": result["predicted_issue"],
        "Urgency Level": result["predicted_urgency"],
        "Confidence": f"{result['urgency_confidence']:.2%}",
        "Product": result["extracted_entities"]["products"] or "Not Detected",
        "Dates Mentioned": ", ".join(result["extracted_entities"]["dates"]) if result["extracted_entities"]["dates"] else "None",
        "Complaint Keywords": ", ".join(result["extracted_entities"]["complaint_keywords"]) if result["extracted_entities"]["complaint_keywords"] else "None"
    }

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="Enter Ticket Text", lines=5),
    outputs=gr.JSON(label="Prediction Results"),
    title="Customer Support Ticket Classifier",
    description="Classify tickets by issue type and urgency level, and extract key entities.",
    examples=[
        ["Payment issue for my SmartWatch V2. I was underbilled for order #29224."],
        ["URGENT! My order from March 15th is 2 weeks late!"],
        ["The PhotoSnap Cam I received is broken and doesn't work."]
    ]
)

iface.launch()
