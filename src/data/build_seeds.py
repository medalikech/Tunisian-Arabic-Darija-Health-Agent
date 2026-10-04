# src/data/build_seeds.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "seeds" / "raw_seeds.txt"
OUT = ROOT / "data" / "seeds" / "all_seeds.jsonl"

counters = {}
with open(IN, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue
        intent, text = [p.strip() for p in line.split("|", 1)]
        counters[intent] = counters.get(intent, 0) + 1
        row = {
            "id": f"{intent}_{counters[intent]:02d}",
            "text": text,
            "intent": intent,
            "source": "human"
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Per-intent counts:")
for k, v in counters.items():
    print(f"  {k}: {v}")