# src/data/debug_jsonl.py
import json
from pathlib import Path

for name in ["data/seeds/all_seeds.jsonl", "data/synthetic/expanded_raw.jsonl"]:
    path = Path(name)
    if not path.exists():
        print(f"{name}: MISSING")
        continue
    print(f"\nChecking {name}:")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                print(f"  Blank line at {i}")
            else:
                try:
                    json.loads(line)
                except Exception as e:
                    print(f"  Bad JSON at line {i}: {line!r}")
    print(f"  Done checking {name}")