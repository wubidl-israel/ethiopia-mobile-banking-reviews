"""
task2_sentiment_theme.py
- Input: all_reviews_processed.csv (from scraping step)
- Output: all_reviews_with_sentiment_and_themes.csv
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import spacy
import re
import json

# ---------- Config ----------
INPUT_CSV = "all_reviews_processed.csv"
OUTPUT_CSV = "all_reviews_with_sentiment_and_themes.csv"
DISTILBERT_BATCH = 32
USE_DISTILBERT = True   # set False to skip heavy transformer model
NUM_KEYWORDS = 30       # extract top-N keywords per bank
NUM_CLUSTERS = 4        # produce 3-5 clusters per bank; default 4
MIN_WORD_LEN = 3
RANDOM_STATE = 42

# ---------- Setup ----------
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ---------- Load data ----------
df = pd.read_csv(INPUT_CSV)
# ensure needed columns exist
assert "review_id" in df.columns or "review_id" in df.columns, "No review_id in CSV"
if "clean_review" not in df.columns:
    df["clean_review"] = df["review_text"].fillna("").astype(str)

# Normalize text function
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

df["clean_review"] = df["clean_review"].apply(clean_text)

# ---------- Sentiment: DistilBERT pipeline (preferred) ----------
if USE_DISTILBERT:
    try:
        clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        def distilbert_sentences(texts):
            # batch inference
            results = []
            for i in range(0, len(texts), DISTILBERT_BATCH):
                batch = texts[i:i+DISTILBERT_BATCH]
                out = clf(batch, truncation=True)
                results.extend(out)
            return results
        # run DistilBERT on non-empty reviews
        texts = df["clean_review"].fillna("").astype(str).tolist()
        tqdm.pandas(desc="DistilBERT sentiment")
        raw_out = distilbert_sentences(texts)
        # raw_out: list of {"label": "POSITIVE"/"NEGATIVE", "score": float}
        df["sentiment_label_model"] = [r["label"].lower() for r in raw_out]
        df["sentiment_score_model"] = [float(r["score"]) if isinstance(r.get("score"), (float,int)) else 0.0 for r in raw_out]
        # Map to polarity range: positive -> +score, negative -> -score
        df["sentiment_score"] = [s if lab=="positive" else -s for lab,s in zip(df["sentiment_label_model"], df["sentiment_score_model"])]
        df["sentiment_label"] = df["sentiment_score"].apply(lambda x: "neutral" if abs(x) < 0.15 else ("positive" if x>0 else "negative"))
    except Exception as e:
        print("DistilBERT failed:", e)
        USE_DISTILBERT = False

# ---------- Fallback: VADER (fast, rule-based) ----------
if not USE_DISTILBERT:
    analyzer = SentimentIntensityAnalyzer()
    def vader_sent(s):
        vs = analyzer.polarity_scores(s)
        # compound in [-1,1]
        return vs["compound"]
    df["sentiment_score"] = df["clean_review"].apply(vader_sent)
    df["sentiment_label"] = df["sentiment_score"].apply(lambda x: "positive" if x>=0.05 else ("negative" if x<=-0.05 else "neutral"))

# ---------- Aggregate by bank and rating (example KPI) ----------
agg = df.groupby(["app_name", "rating"]).agg(
    reviews_count=("review_id","count"),
    avg_sentiment=("sentiment_score","mean"),
).reset_index()
agg.to_csv("kpi_sentiment_by_app_and_rating.csv", index=False)
print("Saved kpi_sentiment_by_app_and_rating.csv")

# ---------- Thematic analysis: TF-IDF keywords per bank ----------

def tokenize_for_tfidf(text):
    # simple spaCy lemmatization and stopword removal
    doc = nlp(text)
    tokens = []
    for t in doc:
        if t.is_alpha and not t.is_stop and len(t.lemma_)>=MIN_WORD_LEN:
            lemma = t.lemma_.lower()
            if lemma not in STOP:
                tokens.append(lemma)
    return " ".join(tokens)

df["tfidf_text"] = df["clean_review"].fillna("").astype(str).apply(tokenize_for_tfidf)

themes_per_bank = {}
df["assigned_theme"] = None

for bank, sub in df.groupby("app_name"):
    texts = sub["tfidf_text"].fillna("").tolist()
    if len(texts) < 10:
        themes_per_bank[bank] = {"clusters": [], "keywords": []}
        df.loc[sub.index, "assigned_theme"] = "other"
        continue

    # TF-IDF
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vect.fit_transform(texts)
    feature_names = np.array(vect.get_feature_names_out())

    # KMeans clustering on TF-IDF vectors for theme discovery
    k = min(NUM_CLUSTERS, max(2, int(len(texts)**0.5)))  # heuristic fallback
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    # collect top terms per cluster
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_keywords = []
    for i in range(k):
        top_terms = feature_names[order_centroids[i, :30]].tolist()
        cluster_keywords.append(top_terms)

    # assign theme names — automatic suggestions (you should manually refine)
    # We'll create short labels from the top keywords of each cluster
    theme_labels = []
    for i, kw in enumerate(cluster_keywords):
        # pick 2-3 representative tokens
        rep = []
        for t in kw:
            # avoid tiny tokens
            if len(t) >= MIN_WORD_LEN:
                rep.append(t)
            if len(rep) >= 3:
                break
        label = "_".join(rep[:3]) if rep else f"theme_{i}"
        theme_labels.append(label)

    # Map each review to cluster label (readable)
    assigned = [theme_labels[l] for l in labels]
    df.loc[sub.index, "assigned_theme"] = assigned

    themes_per_bank[bank] = {
        "clusters": theme_labels,
        "keywords": {label: cluster_keywords[i] for i,label in enumerate(theme_labels)}
    }

# Save themes to JSON for documentation
with open("themes_per_bank.json", "w", encoding="utf-8") as f:
    json.dump(themes_per_bank, f, indent=2)

# ---------- Postprocess: keep important columns and save ----------
save_cols = [
    "review_id", "app_name", "rating", "review_date",
    "clean_review", "sentiment_label", "sentiment_score",
    "assigned_theme", "source"
]
existing = [c for c in save_cols if c in df.columns]
out = df[existing].copy()
out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {OUTPUT_CSV} ({len(out)} rows) and themes_per_bank.json")
