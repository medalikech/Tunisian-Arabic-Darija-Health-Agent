# src/rag/qa_agent.py
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import re
ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data" / "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:14b-instruct"

SIMILARITY_THRESHOLD = 0.35  # tune this after testing real queries

ANSWER_PROMPT = """Tu es un assistant médical pour patients diabétiques tunisiens.
Base ta réponse uniquement sur le contexte ci-dessous. Si l'information n'y
est pas, dis-le clairement et recommande de consulter un pharmacien ou médecin.
N'invente aucune information médicale.

Contexte:
{context}

Question: {question}

Important: réponds uniquement en dialecte tunisien (Derja), écriture arabe. Pas de français.
Réponse en Derja:"""
def embedder():
    return SentenceTransformer(EMBED_MODEL)

def retrieve(question: str, model, collection, k: int = 3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    # Chroma returns distance (lower = more similar) — convert to a rough similarity score
    scored = [(doc, 1 - dist) for doc, dist in zip(docs, distances)]
    return scored

def is_valid_darija_output(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    if re.search(r"[\u4e00-\u9fff\u0400-\u04FF]", text):
        return False
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    total_letters = arabic_chars + latin_chars
    if total_letters == 0:
        return False
    if arabic_chars / total_letters <= 0.5:
        return False
    # NEW: reject if there's a Latin-letter run embedded directly inside
    # an Arabic word (e.g. "المetine", "wireType" fragments) — this
    # catches mid-token corruption the language-ratio check misses.
    if re.search(r"[\u0600-\u06FF][A-Za-z]{2,}|[A-Za-z]{2,}[\u0600-\u06FF]", text):
        return False
    return True

def generate_answer(question: str, context_chunks: list[str], retries: int = 3) -> str:
    context = "\n\n".join(context_chunks)
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }, timeout=300)
            data = resp.json()
            if "response" not in data:
                continue
            answer = clean_answer(data["response"].strip())
            if is_valid_darija_output(answer):
                return answer
            print(f"[WARN] Invalid output on attempt {attempt+1}, retrying: {answer[:60]}...")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Request failed (attempt {attempt+1}/{retries}): {e}")

    return "سمحلي، صار مشكل تقني وما قدرتش نرتب جواب باهي. عاود السؤال من بعد ولا تواصل مع الصيدلي."

def clean_answer(text: str) -> str:
    """Strip stray non-Arabic/French leading tokens the model occasionally emits."""
    import re
    # Remove a leading run of pure-Latin-alphabet word(s) with no Arabic/French
    # medical content markers, which usually indicates a stray artifact token.
    text = re.sub(r"^[A-Za-zÀ-ÿ]+\s+(?=[\u0600-\u06FF])", "", text)
    return text.strip()
def answer_question(question: str) -> dict:
    model = embedder()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
    "diabetes_kb",
    metadata={"hnsw:space": "cosine"})

    scored_chunks = retrieve(question, model, collection)
    best_score = scored_chunks[0][1] if scored_chunks else 0

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answered": False,
            "answer": "ما عنديش معلومة أكيدة على هذا، أحسنلك تسأل الصيدلي ولا الطبيب مباشرة.",
            "best_score": round(best_score, 3),
        }

    top_chunks = [c for c, s in scored_chunks]
    answer = generate_answer(question, top_chunks)
    if not is_valid_darija_output(answer):
        print(f"[DEBUG] FULL rejected text: {answer}")
    return {
        "answered": True,
        "answer": answer,
        "sources": top_chunks,
        "best_score": round(best_score, 3),
    }
    

if __name__ == "__main__":
    test_questions = [
        "وقتاش لازم ناخذ الدواء، قبل ولا بعد الأكلة؟",
        "شنوة الأعراض الجانبية متاع هذا الدواء؟",
        "شنوة أحسن ماكلة في العيد الكبير",  # likely off-corpus, tests the threshold
    ]
    for q in test_questions:
        result = answer_question(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"(answered={result['answered']}, score={result['best_score']})")