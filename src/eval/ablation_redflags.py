# src/eval/ablation_redflags.py
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from nlp.normalize import normalize

SEEDS = ROOT / "data" / "seeds" / "all_seeds.jsonl"
RAW = ROOT / "data" / "synthetic" / "expanded_raw.jsonl"
CLEANED = ROOT / "data" / "synthetic" / "expanded_cleaned.jsonl"
VAL = ROOT / "data" / "processed" / "val.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe(rows):
    by_id = {}
    for r in rows:
        by_id[r.get("id", r.get("text"))] = r
    return list(by_id.values())


def get_eval_groups(eval_rows):
    return {r.get("seed_group", r.get("id")) for r in eval_rows}


def filter_leaked(rows, eval_groups):
    return [r for r in rows if r.get("seed_group", r.get("id")) not in eval_groups]


def train_and_report(train_rows, eval_rows, label):
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

    report = classification_report(y_eval, preds, output_dict=True, zero_division=0)
    red_flag_rows = {k: v for k, v in report.items() if k.startswith("red_flag")}

    print(f"\n=== {label} ===")
    for intent, metrics in red_flag_rows.items():
        print(f"  {intent:25s} f1={metrics['f1-score']:.3f}  support={int(metrics['support'])}")


def main():
    seeds = load_jsonl(SEEDS)
    raw = load_jsonl(RAW)
    cleaned = load_jsonl(CLEANED)
    val_rows = load_jsonl(VAL)

    eval_groups = get_eval_groups(val_rows)
    raw_f = filter_leaked(raw, eval_groups)
    cleaned_f = filter_leaked(cleaned, eval_groups)
    seeds_f = filter_leaked(seeds, eval_groups)

    train_and_report(seeds_f + raw_f, val_rows, "B: Raw LLM expansion")
    train_and_report(seeds_f + cleaned_f, val_rows, "D: Drift-cleaned (final)")


if __name__ == "__main__":
    main()