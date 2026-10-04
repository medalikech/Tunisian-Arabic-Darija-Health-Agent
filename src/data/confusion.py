# src/data/confusion.py
import json
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from nlp.normalize import normalize

def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            row = json.loads(line)
            texts.append(normalize(row["text"]))
            labels.append(row["intent"])
    return texts, labels

pipe = joblib.load(ROOT / "models" / "baseline_tfidf.joblib")
X_val, y_val = load_jsonl(ROOT / "data" / "processed" / "val.jsonl")
preds = pipe.predict(X_val)

labels = sorted(set(y_val))
cm = confusion_matrix(y_val, preds, labels=labels)

print(f"{'':25}" + "".join(f"{l[:8]:>9}" for l in labels))
for i, row in enumerate(cm):
    print(f"{labels[i]:25}" + "".join(f"{v:>9}" for v in row))