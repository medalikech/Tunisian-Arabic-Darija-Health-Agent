# src/rag/build_index.py
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import re

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "knowledge_base"
DB_DIR = ROOT / "data" / "chroma_db"

# Multilingual model: handles French, Arabic, and mixed text reasonably well.
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

def chunk_text(text: str) -> list[str]:
    # Split on the TITRE: markers we used in the knowledge base files,
    # so each chunk is one coherent topic, not an arbitrary character slice.
    chunks = re.split(r"\n(?=TITRE:)", text.strip())
    return [c.strip() for c in chunks if c.strip()]

def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
    "diabetes_kb",
    metadata={"hnsw:space": "cosine"})

    all_chunks, all_ids, all_metadata = [], [], []
    chunk_id = 0

    for file in KB_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadata.append({"source": file.name})
            chunk_id += 1

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )
    print(f"Indexed {len(all_chunks)} chunks into {DB_DIR}")

if __name__ == "__main__":
    main()