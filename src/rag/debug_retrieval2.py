# src/rag/debug_retrieval2.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import embedder, get_collection, retrieve

model = embedder()
collection = get_collection()

question = "\u0648\u0642\u062a\u0627\u0634 \u0644\u0627\u0632\u0645 \u0646\u0627\u062e\u0630 \u0627\u0644\u062f\u0648\u0627\u0621\u061f"
scored = retrieve(question, model, collection)

for doc, score, meta in scored:
    print(f"score={score:.3f} | source={meta['source']}")
    print(f"  {doc[:60]}...")