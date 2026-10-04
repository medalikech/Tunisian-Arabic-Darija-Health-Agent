# src/data/review_cli.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def review(in_path: Path, out_path: Path):
    with open(in_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    kept = []
    keep_all_rest = False

    for i, r in enumerate(rows):
        if keep_all_rest:
            r["reviewed"] = False  # mark as NOT manually reviewed, kept in bulk
            kept.append(r)
            continue

        print(f"\n[{i+1}/{len(rows)}]")
        print(f"Intent: {r['intent']}")
        print(f"Text:   {r['text']}")
        choice = input("[k]eep / [d]rop / [e]dit / [K]eep all rest / [q]uit: ").strip()

        if choice == "q":
            break
        elif choice == "K":
            keep_all_rest = True
            r["reviewed"] = False
            kept.append(r)
        elif choice == "d":
            continue
        elif choice == "e":
            r["text"] = input("New text: ").strip()
            r["reviewed"] = True
            kept.append(r)
        else:  # lowercase 'k' or anything else defaults to keep
            r["reviewed"] = True
            kept.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    unreviewed = sum(1 for r in kept if not r.get("reviewed"))
    print(f"\nKept {len(kept)}/{len(rows)} ({unreviewed} auto-kept without manual review)")

if __name__ == "__main__":
    review(
        ROOT / "data" / "synthetic" / "expanded_raw.jsonl",
        ROOT / "data" / "synthetic" / "expanded_reviewed.jsonl",
    )