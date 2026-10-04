# src/data/clean_drift.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "synthetic" / "expanded_reviewed.jsonl"
OUT = ROOT / "data" / "synthetic" / "expanded_cleaned.jsonl"

SUSPECT_WORDS = [
    "بدي", "شو ", "شو؟", "ليش", "هيك", "منيح", "كتير",
    "هالزمن", "هالشي", "هادشي", "بحالي", "زي العقارب"
]

kept, dropped = 0, 0
with open(IN, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        row = json.loads(line)
        if any(w in row["text"] for w in SUSPECT_WORDS):
            dropped += 1
            continue
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1

print(f"Kept {kept}, dropped {dropped}")