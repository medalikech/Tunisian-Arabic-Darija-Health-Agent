# src/rag/debug_index_metadata.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import chromadb
from rag.qa_agent import DB_DIR

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection("diabetes_kb")
results = collection.get()

for doc, meta in zip(results["documents"], results["metadatas"]):
    fallback = meta.get("darija_fallback", "")
    print(f"[{meta['source']}] fallback_len={len(fallback)} | {doc[:40]}...")