# src/nlp/baseline.py
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nlp.normalize import normalize

def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8-sig") as f:   # changed from "utf-8"
        for line in f:
            row = json.loads(line)
            texts.append(normalize(row["text"]))
            labels.append(row["intent"])
    return texts, labels

def train_baseline():
    X_train, y_train = load_jsonl("data/processed/train.jsonl")
    X_val, y_val = load_jsonl("data/processed/val.jsonl")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    print(classification_report(y_val, preds))
    joblib.dump(pipe, "models/baseline_tfidf.joblib")

if __name__ == "__main__":
    train_baseline()