# src/data/check_split.py
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

for name in ["train", "val", "test"]:
    path = ROOT / "data" / "processed" / f"{name}.jsonl"
    c = Counter(json.loads(l)["intent"] for l in open(path, encoding="utf-8"))
    print(f"--- {name} ---")
    for intent, count in sorted(c.items()):
        print(f"  {intent}: {count}")