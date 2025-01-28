"""
End-to-end NLP Analysis Tool

Steps:
1. Preprocess text
2. Train text classifier (example: NLTK Naive Bayes)
3. Perform sentiment analysis (VADER + spaCy example)
4. Provide final insights/outputs
"""

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string

# 1. Data ingestion (here we just define documents inline)
documents = [
    "I love this new phone! It’s so fast and the camera is amazing.",
    "This service is terrible. I am never going to use it again.",
    "The movie was okay, but I expected more action.",
    "Wonderful experience! The staff was friendly, and the food was delicious.",
    "I am quite disappointed with the product quality.",
]

# 2. Preprocessing
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [t for t in tokens if t not in stop_words and t not in punctuations]
    return filtered_tokens

preprocessed_docs = [preprocess_text(doc) for doc in documents]

# 3. Simple sentiment analysis using NLTK's VADER
sid = SentimentIntensityAnalyzer()
for idx, doc in enumerate(documents):
    scores = sid.polarity_scores(doc)
    print(f"Document {idx+1}: '{doc}' -> {scores}")

# 4. Additional spaCy-based analysis (named entities)
nlp = spacy.load("en_core_web_sm")
for idx, doc in enumerate(documents):
    spacy_doc = nlp(doc)
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
    print(f"Document {idx+1}: Named Entities -> {entities}")

print("\n=== NLP Analysis Complete ===")
