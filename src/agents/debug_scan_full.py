# src/agents/debug_scan_full.py
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

files_to_check = [
    ROOT / "data" / "seeds" / "all_seeds.jsonl",
    ROOT / "data" / "synthetic" / "expanded_reviewed.jsonl",
    ROOT / "data" / "processed" / "all_labeled.jsonl",
]

for path in files_to_check:
    garbled_count = 0
    total = 0
    first_garbled_example = None
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            total += 1
            if any(0x80 <= ord(c) <= 0xFF for c in text):
                garbled_count += 1
                if first_garbled_example is None:
                    first_garbled_example = text
    print(f"{path.name}: {garbled_count}/{total} garbled")
    if first_garbled_example:
        print(f"  example: {first_garbled_example[:50]!r}")