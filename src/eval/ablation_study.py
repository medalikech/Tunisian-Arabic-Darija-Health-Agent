# src/eval/ablation_study.py
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from nlp.normalize import normalize

SEEDS = ROOT / "data" / "seeds" / "all_seeds.jsonl"
RAW = ROOT / "data" / "synthetic" / "expanded_raw.jsonl"
REVIEWED = ROOT / "data" / "synthetic" / "expanded_reviewed.jsonl"
CLEANED = ROOT / "data" / "synthetic" / "expanded_cleaned.jsonl"

# Use the existing, final val/test split for ALL stages so the evaluation
# set stays constant — only the TRAINING data changes between stages.
VAL = ROOT / "data" / "processed" / "val.jsonl"
TEST = ROOT / "data" / "processed" / "test.jsonl"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe(rows):
    """Keep the last occurrence of each id, since files were appended to
    across multiple rounds and may contain repeated/updated entries."""
    by_id = {}
    for r in rows:
        key = r.get("id", r.get("text"))
        by_id[key] = r
    return list(by_id.values())

def get_eval_groups(eval_rows):
    """Collect every seed_group/id present in the evaluation set, so we can
    exclude any training example derived from the same seed."""
    groups = set()
    for r in eval_rows:
        groups.add(r.get("seed_group", r.get("id")))
    return groups

def filter_leaked(train_rows, eval_groups):
    """Remove any training row whose seed_group/id also appears in eval."""
    return [r for r in train_rows if r.get("seed_group", r.get("id")) not in eval_groups]
def train_and_eval(train_rows, eval_rows, label):
    train_rows = dedupe(train_rows)
    X_train = [normalize(r["text"]) for r in train_rows]
    y_train = [r["intent"] for r in train_rows]
    X_eval = [normalize(r["text"]) for r in eval_rows]
    y_eval = [r["intent"] for r in eval_rows]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_eval)

    macro_f1 = f1_score(y_eval, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_eval, preds)

    print(f"{label:35s} | train_n={len(train_rows):5d} | macro_f1={macro_f1:.3f} | acc={acc:.3f}")
    return {"stage": label, "train_n": len(train_rows), "macro_f1": round(macro_f1, 3), "accuracy": round(acc, 3)}


def main():
    seeds = load_jsonl(SEEDS)
    raw = load_jsonl(RAW)
    reviewed = load_jsonl(REVIEWED)
    cleaned = load_jsonl(CLEANED)
    val_rows = load_jsonl(VAL)
    test_rows = load_jsonl(TEST)

    eval_groups = get_eval_groups(val_rows) | get_eval_groups(test_rows)

    seeds = filter_leaked(seeds, eval_groups)
    raw = filter_leaked(raw, eval_groups)
    reviewed = filter_leaked(reviewed, eval_groups)
    cleaned = filter_leaked(cleaned, eval_groups)

    print(f"After removing leaked seed groups: seeds={len(seeds)}, raw={len(raw)}, "
          f"reviewed={len(reviewed)}, cleaned={len(cleaned)}")

    print(f"\nEvaluating on VAL set ({len(val_rows)} examples) for all stages:\n")

    results = []
    results.append(train_and_eval(seeds, val_rows, "A: Seeds only"))
    results.append(train_and_eval(seeds + raw, val_rows, "B: + Raw LLM expansion"))
    results.append(train_and_eval(seeds + reviewed, val_rows, "C: + Reviewed expansion"))
    results.append(train_and_eval(seeds + cleaned, val_rows, "D: + Drift-cleaned (final)"))

    print(f"\nSame stages evaluated on TEST set ({len(test_rows)} examples):\n")
    for label, data in [
        ("A: Seeds only", seeds),
        ("B: + Raw LLM expansion", seeds + raw),
        ("C: + Reviewed expansion", seeds + reviewed),
        ("D: + Drift-cleaned (final)", seeds + cleaned),
    ]:
        train_and_eval(data, test_rows, label)

    out_path = ROOT / "data" / "eval" / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()