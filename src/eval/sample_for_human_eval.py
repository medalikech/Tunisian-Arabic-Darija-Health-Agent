# src/eval/sample_for_human_eval.py
import sys, random, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import sqlite3
from patient_storage.conversation_log import DB_PATH
from patient_storage.data_crypto import decrypt_text

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT * FROM messages WHERE raw_text != '' ORDER BY RANDOM() LIMIT 300"
).fetchall()
conn.close()

# Stratify: try to get a spread across routes, not just whatever's most common
by_route = {}
for r in rows:
    by_route.setdefault(r["route"], []).append(r)

sample = []
per_route = 5  # aim for ~5 examples per route type, adjust based on what exists
for route, items in by_route.items():
    sample.extend(random.sample(items, min(per_route, len(items))))

random.shuffle(sample)

out_path = ROOT / "data" / "eval" / "human_eval_sheet.csv"
with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "user_message", "bot_response", "route",
        "naturalness_1to5", "correctness_1to5", "safety_1to5", "notes"
    ])
    for i, r in enumerate(sample, 1):
        writer.writerow([
            i,
            decrypt_text(r["raw_text"]),
            decrypt_text(r["response"]),
            r["route"],
            "",
            "",
            "",
            "",
        ])

print(f"Wrote {len(sample)} examples to {out_path}")
print("Route breakdown:", {route: len(items) for route, items in by_route.items()})