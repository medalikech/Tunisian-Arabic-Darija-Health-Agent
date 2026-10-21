# src/rag/debug_retrieval.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import embedder, get_collection, retrieve

model = embedder()
collection = get_collection()

question = "وقتاش لازم ناخذ الدواء"
scored = retrieve(question, model, collection)

for doc, score, meta in scored:
    print(f"score={score:.3f} | source={meta['source']} | fallback_len={len(meta.get('darija_fallback',''))}")
    print(f"  {doc[:60]}...")