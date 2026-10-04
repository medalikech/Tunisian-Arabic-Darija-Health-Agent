# src/agents/debug_vectorizer.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import joblib
from nlp.normalize import normalize

pipe = joblib.load(ROOT / "models" / "baseline_tfidf.joblib")
tfidf = pipe.named_steps["tfidf"]

msg1 = normalize("وقتاش لازم ناخذ الدواء")
msg2 = normalize("سلام كيفاش الحال")

vec1 = tfidf.transform([msg1])
vec2 = tfidf.transform([msg2])

print("Vocab size:", len(tfidf.vocabulary_))
print("msg1 nonzero features:", vec1.nnz)
print("msg2 nonzero features:", vec2.nnz)
print("Are the two vectors identical?", (vec1 != vec2).nnz == 0)