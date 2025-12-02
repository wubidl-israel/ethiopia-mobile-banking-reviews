# src/preprocess_reviews.py
import pandas as pd, glob, os
from dateutil import parser

RAW_DIR = "data/raw/"
OUT_DIR = "data/processed/"
os.makedirs(OUT_DIR, exist_ok=True)

def load_all_raw():
    files = glob.glob(RAW_DIR + "*_raw.csv")
    dfs = [pd.read_csv(f, encoding='utf-8') for f in files]
    return pd.concat(dfs, ignore_index=True)

def normalize_date(d):
    try:
        return pd.to_datetime(d).date().isoformat()
    except Exception:
        try:
            return parser.parse(str(d)).date().isoformat()
        except Exception:
            return pd.NaT

def preprocess():
    df = load_all_raw()
    df = df.rename(columns=lambda x: x.strip().lower())
    want = ['review','rating','date_raw','bank','source']
    df = df[[c for c in want if c in df.columns]]
    df['review'] = df['review'].astype(str).str.strip()
    df = df[df['review'] != ""].copy()
    df = df.drop_duplicates(subset=['review','rating','bank'])
    df['date'] = df['date_raw'].apply(normalize_date)
    df = df.dropna(subset=['rating'])  # keep rows with ratings
    out_path = OUT_DIR + "reviews_clean.csv"
    df[['review','rating','date','bank','source']].to_csv(out_path, index=False, encoding='utf-8')
    print("Saved cleaned CSV:", out_path)
    return df

if __name__ == "__main__":
    preprocess()
