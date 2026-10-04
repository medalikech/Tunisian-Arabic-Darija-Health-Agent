# src/agents/debug_pipeline_stages.py
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

files_to_check = [
    ROOT / "data" / "seeds" / "all_seeds.jsonl",
    ROOT / "data" / "synthetic" / "expanded_reviewed.jsonl",
    ROOT / "data" / "processed" / "all_labeled.jsonl",
    ROOT / "data" / "processed" / "train.jsonl",
]

for path in files_to_check:
    if not path.exists():
        print(f"{path.name}: MISSING")
        continue
    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
    row = json.loads(first_line)
    text = row["text"]
    is_garbled = any(0x80 <= ord(c) <= 0xFF for c in text)
    print(f"{path.name}: {'GARBLED' if is_garbled else 'OK'} | {text[:40]!r}")