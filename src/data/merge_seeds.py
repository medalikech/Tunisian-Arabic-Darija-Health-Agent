# src/data/merge_seeds.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

files = [
    ROOT / "data" / "seeds" / "all_seeds.jsonl",
    ROOT / "data" / "synthetic" / "expanded_cleaned.jsonl",  # or expanded_reviewed.jsonl, whichever you used last
]
out_path = ROOT / "data" / "processed" / "all_labeled.jsonl"

with open(out_path, "w", encoding="utf-8") as out:
    for fpath in files:
        with open(fpath, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + "\n")

print(f"Merged into {out_path}")