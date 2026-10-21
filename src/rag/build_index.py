# src/rag/build_index.py
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
import re

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "knowledge_base"
DB_DIR = ROOT / "data" / "chroma_db"

# Multilingual model: handles French, Arabic, and mixed text reasonably well.
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

def chunk_text(text: str) -> list[dict]:
    raw_chunks = re.split(r"\n(?=TITRE:)", text.strip())
    parsed = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if "DARIJA:" in chunk:
            french_part, darija_part = chunk.split("DARIJA:", 1)
            parsed.append({
                "text": french_part.strip(),
                "darija_fallback": darija_part.strip(),
            })
        else:
            parsed.append({"text": chunk, "darija_fallback": ""})
    return parsed

def index_knowledge_base(collection, model, kb_dir: Path = KB_DIR) -> int:
    all_chunks, embedding_texts, all_ids, all_metadata = [], [], [], []

    for file in sorted(kb_dir.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk["text"])
            embedding_texts.append(chunk["darija_fallback"] or chunk["text"])
            stable_key = f"{file.name}:{chunk_index}".encode("utf-8")
            all_ids.append(f"chunk_{hashlib.sha256(stable_key).hexdigest()}")
            all_metadata.append({"source": file.name, "darija_fallback": chunk["darija_fallback"]})

    if not all_chunks:
        raise ValueError(f"No knowledge-base chunks found in {kb_dir}.")

    embeddings = model.encode(embedding_texts, show_progress_bar=True).tolist()
    existing_ids = set(collection.get()["ids"])
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )
    stale_ids = existing_ids.difference(all_ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))

    return len(all_chunks)


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        "diabetes_kb",
        metadata={"hnsw:space": "cosine"},
    )

    count = index_knowledge_base(collection, model)
    print(f"Indexed {count} chunks into {DB_DIR}")

if __name__ == "__main__":
    main()