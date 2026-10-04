# src/data/split.py
import json, random
from collections import defaultdict

def group_split(path: str, train=0.7, val=0.15, test=0.15, seed=42):
    by_group = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_group[row["seed_group"]].append(row)

    groups = list(by_group.keys())
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_train, n_val = int(n * train), int(n * val)

    splits = {"train": [], "val": [], "test": []}
    for i, g in enumerate(groups):
        bucket = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        splits[bucket].extend(by_group[g])

    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", errors="strict") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples from {sum(1 for g in groups[:n_train] if True)} groups")