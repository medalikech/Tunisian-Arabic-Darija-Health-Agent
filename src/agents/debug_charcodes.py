# src/agents/debug_charcodes.py
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

# Grab one real training example to compare against
with open(ROOT / "data" / "processed" / "train.jsonl", encoding="utf-8-sig") as f:
    first_row = json.loads(f.readline())

print("Training example text:", repr(first_row["text"]))
print("Char codes:", [hex(ord(c)) for c in first_row["text"][:10]])

test = "سلام كيفاش الحال"
print("\nTest message text:", repr(test))
print("Char codes:", [hex(ord(c)) for c in test[:10]])