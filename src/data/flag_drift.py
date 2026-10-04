# src/data/flag_drift.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUSPECT_WORDS = ["بدي", "شو ", "ليش", "هيك", "منيح", "كتير"]  # common Levantine/Gulf markers

def flag(path):
    flagged = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if any(w in row["text"] for w in SUSPECT_WORDS):
                flagged.append(row)
    print(f"Flagged {len(flagged)} rows for review:")
    for r in flagged[:50]:
        print(f"  [{r['intent']}] {r['text']}")

if __name__ == "__main__":
    flag(ROOT / "data" / "synthetic" / "expanded_reviewed.jsonl")