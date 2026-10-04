# src/agents/debug_single.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import joblib
from nlp.normalize import normalize

CLASSIFIER = joblib.load(ROOT / "models" / "baseline_tfidf.joblib")

msg = "mridh"
normalized = normalize(msg)
probs = CLASSIFIER.predict_proba([normalized])[0]
classes = CLASSIFIER.classes_
top5 = sorted(zip(classes, probs), key=lambda x: -x[1])[:5]
print(f"Input: {msg}")
for cls, p in top5:
    print(f"  {cls}: {p:.3f}")