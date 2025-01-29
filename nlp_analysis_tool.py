"""
NLP Analysis Tool
=================
This script demonstrates how to:
1. Read labeled data (CSV) and unlabeled data (TXT)
2. Preprocess and tokenize text
3. Classify text using NLTK (Naive Bayes)
4. Analyze sentiment with NLTK VADER
5. Extract named entities using spaCy

File Requirements:
-----------------
- "path/to/labeled_data.csv": Must have at least two columns: "text" and "label"
  Example:
      text,label
      "I love this new phone!","positive"
      "This service is terrible.","negative"

- "path/to/unlabeled_data.txt": Each line is one document to process.

Instructions:
-------------
1. Update the file paths below to your actual data files.
2. pip install nltk spacy pandas
3. python -m spacy download en_core_web_sm
4. python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
5. python nlp_analysis_tool.py

The script will print:
- Classifier accuracy on a small test set.
- Classification results for new, unseen texts.
- VADER sentiment scores for the unlabeled documents.
- Named entities extracted by spaCy for the unlabeled documents.
"""

import nltk
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string

# ========== 1. Data Reading ==========

# Adjust these paths to point to your actual data files
LABELED_DATA_CSV = "data/labeled_data.csv"
UNLABELED_DATA_TXT = "data/unlabeled_data.txt"

# Read labeled data from CSV (must have columns: text, label)
labeled_df = pd.read_csv(LABELED_DATA_CSV)
# Convert DataFrame rows to a list of [text, label]
labeled_data = labeled_df.values.tolist()

# Read unlabeled data from TXT (each line is one document)
with open(UNLABELED_DATA_TXT, "r", encoding="utf-8") as f:
    unlabeled_documents = [line.strip() for line in f if line.strip()]


# ========== 2. Preprocessing ==========

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def preprocess_text(text):
    """
    1) Lowercase
    2) Tokenize
    3) Remove stop words & punctuation
    """
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [t for t in tokens if t not in stop_words and t not in punctuations]
    return filtered_tokens


# ========== 3. Text Classification with NLTK ==========

def document_features(doc_tokens):
    """
    Convert a list of tokens into a dictionary {word: True} 
    for a simple 'bag-of-words' approach used by NaiveBayesClassifier.
    """
    return {word: True for word in doc_tokens}

def train_text_classifier(labeled_dataset):
    """
    Train a Naive Bayes classifier on labeled data (list of [text, label]).
    We'll do a simple train/test split for demonstration.
    """
    # Convert each labeled sample into (features, label)
    featuresets = []
    for row in labeled_dataset:
        # row is something like ["I love this new phone!", "positive"]
        text, label = row[0], row[1]
        tokens = preprocess_text(text)
        featuresets.append((document_features(tokens), label))
    
    # Simple train/test split (80/20 split in this example)
    split_index = int(len(featuresets) * 0.8)
    train_set = featuresets[:split_index]
    test_set = featuresets[split_index:]
    
    # Train
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    # Evaluate
    if test_set:
        accuracy = nltk.classify.accuracy(classifier, test_set)
        print(f"[Classifier] Test set accuracy: {accuracy:.2f}")
        classifier.show_most_informative_features()
    else:
        print("[Classifier] Warning: No test set was created (too few samples).")
    
    return classifier

def classify_new_texts(classifier, texts):
    """
    Classify a list of new, unseen texts with a trained classifier.
    Returns a list of (text, predicted_label).
    """
    results = []
    for text in texts:
        tokens = preprocess_text(text)
        feats = document_features(tokens)
        label = classifier.classify(feats)
        results.append((text, label))
    return results


# ========== 4. Sentiment Analysis with NLTK VADER ==========

def analyze_sentiment_vader(documents):
    """
    Use NLTK's VADER to compute sentiment polarity scores for each doc.
    Returns a list of (document, {neg, neu, pos, compound}).
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_results = []
    for doc in documents:
        scores = sid.polarity_scores(doc)
        sentiment_results.append((doc, scores))
    return sentiment_results


# ========== 5. Named Entity Extraction with spaCy ==========

def extract_named_entities_spacy(documents):
    """
    Use spaCy to extract named entities from each document.
    Returns a list of (document, [(entity_text, entity_label), ...]).
    """
    nlp = spacy.load("en_core_web_sm")
    entity_results = []
    
    for doc in documents:
        spacy_doc = nlp(doc)
        entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
        entity_results.append((doc, entities))
    
    return entity_results


# ========== 6. Main Flow (Demonstration) ==========

def main():
    # -- A) Train & Evaluate a Text Classifier --
    classifier = train_text_classifier(labeled_data)
    
    # -- B) Classify New Texts (Example) --
    new_texts = [
        "The product arrived quickly, but the quality was mediocre.",
        "Absolutely loved it! Best purchase I've made this year.",
        "It was just alright. Not bad, not great."
    ]
    classification_results = classify_new_texts(classifier, new_texts)
    print("\n[Classification on New Texts]")
    for text, label in classification_results:
        print(f"Text: '{text}' -> Predicted Label: {label}")
    
    # -- C) Perform Sentiment Analysis on Unlabeled Data --
    sentiment_results = analyze_sentiment_vader(unlabeled_documents)
    print("\n[VADER Sentiment Analysis on Unlabeled Documents]")
    for doc, scores in sentiment_results:
        print(f"Text: '{doc}'\nScores: {scores}\n")
    
    # -- D) Named Entity Extraction with spaCy --
    entity_results = extract_named_entities_spacy(unlabeled_documents)
    print("[Named Entity Extraction]")
    for doc, ents in entity_results:
        print(f"Text: '{doc}'")
        if ents:
            for text, label in ents:
                print(f"  - Entity: {text} | Label: {label}")
        else:
            print("  - No named entities found.")
        print()

if __name__ == "__main__":
    main()
