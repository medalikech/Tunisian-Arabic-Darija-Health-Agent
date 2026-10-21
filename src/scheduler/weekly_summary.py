# src/scheduler/weekly_summary.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.adherence_log import get_adherence_stats, init_adherence_db
from patient_storage.conversation_log import DB_PATH
import sqlite3

def get_all_known_patients() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT DISTINCT patient_id FROM messages").fetchall()
    conn.close()
    return [r[0] for r in rows]

def format_summary(patient_id: str, stats: dict) -> str:
    if stats["total"] == 0:
        return f"{patient_id}: \u0644\u0627 \u062a\u0648\u062c\u062f \u0628\u064a\u0627\u0646\u0627\u062a \u0643\u0627\u0641\u064a\u0629 \u0647\u0630\u0627 \u0627\u0644\u0623\u0633\u0628\u0648\u0639."

    rate_pct = int(stats["adherence_rate"] * 100)
    flag = ""
    if stats["adherence_rate"] < 0.7:
        flag = "  \u26a0\ufe0f \u0645\u0639\u062f\u0644 \u0627\u0644\u0627\u0644\u062a\u0632\u0627\u0645 \u0645\u0646\u062e\u0641\u0636 \u2014 \u064a\u0633\u062a\u062d\u0642 \u0645\u062a\u0627\u0628\u0639\u0629"

    return (
        f"{patient_id}: {stats['taken']}/{stats['total']} \u062c\u0631\u0639\u0629 "
        f"({rate_pct}%){flag}"
    )

def run_weekly_summary(days: int = 7):
    init_adherence_db()
    patients = get_all_known_patients()

    print(f"=== Weekly Adherence Summary ({days} days) ===\n")
    if not patients:
        print("No patients found in the log yet.")
        return

    low_adherence_patients = []
    for patient_id in patients:
        stats = get_adherence_stats(patient_id, days=days)
        print(format_summary(patient_id, stats))
        if stats["total"] > 0 and stats["adherence_rate"] < 0.7:
            low_adherence_patients.append(patient_id)

    if low_adherence_patients:
        print(f"\n{len(low_adherence_patients)} patient(s) below 70% adherence this week:")
        for p in low_adherence_patients:
            print(f"  - {p}")

if __name__ == "__main__":
    run_weekly_summary()