# src/data/split.py
import json, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

def group_split(path: Path, train=0.7, val=0.15, test=0.15, seed=42):
    # Group rows by seed_group (fallback to id for original human seeds)
    by_group = defaultdict(list)
    for row in (json.loads(l) for l in open(path, encoding="utf-8-sig")):
        key = row.get("seed_group", row.get("id"))
        by_group[key].append(row)

    # Figure out which intent each group belongs to
    group_intent = {g: rows[0]["intent"] for g, rows in by_group.items()}

    # Bucket groups by intent first
    groups_by_intent = defaultdict(list)
    for g, intent in group_intent.items():
        groups_by_intent[intent].append(g)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for intent, groups in groups_by_intent.items():
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, int(n * train))
        n_val = max(1, int(n * val)) if n > 2 else 0

        for i, g in enumerate(groups):
            if i < n_train:
                bucket = "train"
            elif i < n_train + n_val:
                bucket = "val"
            else:
                bucket = "test"
            splits[bucket].extend(by_group[g])

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples")

if __name__ == "__main__":
    group_split(ROOT / "data" / "processed" / "all_labeled.jsonl")