# src/data/expand.py
import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b-instruct"

EXPANSION_PROMPT = """You are generating training data for a Tunisian Arabic/Darija
medical chatbot intent classifier.

Given this seed message (intent: {intent}):
"{seed_text}"

Generate {n} variants that express the SAME intent, varying:
- script (Arabic script vs Arabizi, using digits 3,7,9,5,2 for the Arabic letters ain, ha, qaf, kha, hamza)
- formality (casual street Darija vs more formal/MSA-leaning)
- some with French code-switching (French words mixed into Darija)
- some with typos or no punctuation
- some very short, some more detailed

IMPORTANT: Use TUNISIAN dialect specifically, not Levantine, Egyptian, or Gulf Arabic.

Do NOT change the intent. Do NOT add new intents or medical claims.
Return ONLY a JSON array of strings, nothing else. No markdown, no explanation.
"""

def expand_seed(seed_text, intent, n=10, retries=3):
    prompt = EXPANSION_PROMPT.format(intent=intent, seed_text=seed_text, n=n)

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.9}
            }, timeout=120)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)
            continue

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"[WARN] Non-JSON HTTP response (attempt {attempt+1}/{retries}), status {resp.status_code}")
            time.sleep(2)
            continue

        if "response" not in data:
            print(f"[WARN] No 'response' key (attempt {attempt+1}/{retries}): {data}")
            time.sleep(2)
            continue

        raw = data["response"].strip()
        raw = raw.strip("```json").strip("```").strip()
        try:
            variants = json.loads(raw)
            assert isinstance(variants, list)
            return variants
        except (json.JSONDecodeError, AssertionError):
            print(f"[WARN] Bad JSON in model output for seed: {seed_text[:40]}...")
            return []

    print(f"[ERROR] Giving up on seed after {retries} attempts: {seed_text[:40]}")
    return []
