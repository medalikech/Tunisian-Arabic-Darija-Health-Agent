# src/agents/debug_classify.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import joblib
from nlp.normalize import normalize

CLASSIFIER = joblib.load(ROOT / "models" / "baseline_tfidf.joblib")

test_messages = [
    "وقتاش لازم ناخذ الدواء",
    "نحس بالضيق من هذا المرض",
    "سلام كيفاش الحال",
]

for msg in test_messages:
    normalized = normalize(msg)
    probs = CLASSIFIER.predict_proba([normalized])[0]
    classes = CLASSIFIER.classes_
    top3 = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]
    print(f"\nInput: {msg}")
    for cls, p in top3:
        print(f"  {cls}: {p:.3f}")