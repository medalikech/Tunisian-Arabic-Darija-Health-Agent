# src/data/run_expansion.py
import json, time
from pathlib import Path
from expand import expand_seed

ROOT = Path(__file__).resolve().parents[2]

def run(seeds_path: Path, out_path: Path, n_per_seed: int = 10):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(seeds_path, encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    # Resume support: skip seeds already processed in a previous run
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done_ids.add(row["seed_group"])
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as out:
        for i, seed in enumerate(seeds):
            if seed["id"] in done_ids:
                continue  # already expanded in a previous run

            variants = expand_seed(seed["text"], seed["intent"], n=n_per_seed)
            print(f"[{i+1}/{len(seeds)}] {seed['intent']}: {len(variants)} variants")

            for j, v in enumerate(variants):
                row = {
                    "id": f"{seed['id']}_v{j}",
                    "text": v,
                    "intent": seed["intent"],
                    "seed_group": seed["id"],
                    "source": "synthetic",
                    "reviewed": False
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()   # write immediately, don't lose progress if it crashes
            time.sleep(0.2)

if __name__ == "__main__":
    run(
        ROOT / "data" / "seeds" / "all_seeds.jsonl",
        ROOT / "data" / "synthetic" / "expanded_raw.jsonl",
    )