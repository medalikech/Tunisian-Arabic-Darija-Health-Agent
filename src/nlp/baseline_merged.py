# src/nlp/baseline_merged.py
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from pathlib import Path
import sys
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from nlp.normalize import normalize

RED_FLAG_MERGE = {
    "red_flag_chest_pain": "red_flag_emergency",
    "red_flag_breathing": "red_flag_emergency",
    "red_flag_hypoglycemia": "red_flag_emergency",
    "red_flag_overdose": "red_flag_emergency",
    "red_flag_suicidal": "red_flag_emergency",
}

def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            row = json.loads(line)
            texts.append(normalize(row["text"]))
            labels.append(RED_FLAG_MERGE.get(row["intent"], row["intent"]))
    return texts, labels

X_train, y_train = load_jsonl(ROOT / "data" / "processed" / "train.jsonl")
X_val, y_val = load_jsonl(ROOT / "data" / "processed" / "val.jsonl")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_val)
print(classification_report(y_val, preds))
joblib.dump(pipe, ROOT / "models" / "baseline_merged.joblib")
print("Saved to models/baseline_merged.joblib")