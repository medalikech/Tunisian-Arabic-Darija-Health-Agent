# src/data/review_cli.py
import json

def review(in_path: str, out_path: str):
    with open(in_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    kept = []
    for r in rows:
        print(f"\nIntent: {r['intent']}")
        print(f"Text:   {r['text']}")
        choice = input("[k]eep / [d]rop / [e]dit / [q]uit: ").strip().lower()
        if choice == "q":
            break
        elif choice == "d":
            continue
        elif choice == "e":
            r["text"] = input("New text: ").strip()
            r["reviewed"] = True
            kept.append(r)
        else:
            r["reviewed"] = True
            kept.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nKept {len(kept)}/{len(rows)}")

if __name__ == "__main__":
    review("data/synthetic/expanded_raw.jsonl", "data/synthetic/expanded_reviewed.jsonl")