# src/rag/check_index.py
import chromadb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data" / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection("diabetes_kb")

print("Total chunks:", collection.count())
results = collection.get()
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"[{meta['source']}] {doc[:60]}...")