# src/eval/export_transcripts.py
import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.conversation_log import DB_PATH
import sqlite3

OUT_PATH = ROOT / "data" / "eval" / "transcripts.md"

def export_all(limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM messages ORDER BY timestamp ASC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Conversation Transcripts\n\n")
        f.write(f"Exported {len(rows)} interactions.\n\n")

        current_patient = None
        for row in rows:
            if row["patient_id"] != current_patient:
                current_patient = row["patient_id"]
                f.write(f"\n## Patient: {current_patient}\n\n")

            f.write(f"**User:** {row['raw_text']}\n\n")
            f.write(f"**Route:** `{row['route']}`")
            if row["intent"]:
                f.write(f" | **Intent:** `{row['intent']}`")
            if row["confidence"] is not None:
                f.write(f" | **Confidence:** {row['confidence']}")
            if row["matched_red_flags"]:
                flags = json.loads(row["matched_red_flags"])
                f.write(f" | **Red flags:** {', '.join(flags)}")
            f.write("\n\n")
            f.write(f"**Assistant:** {row['response']}\n\n")
            f.write("---\n\n")

    print(f"Exported {len(rows)} interactions to {OUT_PATH}")
    # Quick summary stats, useful for your report
    routes = [r["route"] for r in rows]
    from collections import Counter
    print("\nRoute breakdown:")
    for route, count in Counter(routes).most_common():
        print(f"  {route}: {count}")

if __name__ == "__main__":
    export_all()