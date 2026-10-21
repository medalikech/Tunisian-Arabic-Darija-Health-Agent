# src/eval/aggregate_human_eval.py
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "eval" / "human_eval_sheet.csv"

rows = []
with open(IN_PATH, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

scored = [r for r in rows if r["naturalness_1to5"].strip()]
print(f"Scored {len(scored)}/{len(rows)} examples\n")

if scored:
    for metric in ["naturalness_1to5", "correctness_1to5", "safety_1to5"]:
        values = [float(r[metric]) for r in scored if r[metric].strip()]
        if values:
            print(f"{metric}: avg={sum(values)/len(values):.2f}  n={len(values)}")

    by_route = defaultdict(list)
    for r in scored:
        by_route[r["route"]].append(float(r["correctness_1to5"]))
    print("\nCorrectness by route:")
    for route, vals in by_route.items():
        print(f"  {route}: avg={sum(vals)/len(vals):.2f}  n={len(vals)}")