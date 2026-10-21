# src/scheduler/clinician_view.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.escalation_log import (
    init_escalation_db,
    get_pending_escalations,
    resolve_escalation,
    get_escalation_counts_by_patient,
)

def main():
    init_escalation_db()
    pending = get_pending_escalations()

    print(f"=== Pending Escalations ({len(pending)}) ===\n")
    for e in pending:
        print(f"[{e['id']}] {e['severity'].upper()} | {e['patient_id']} | {e['timestamp']}")
        print(f"    \"{e['raw_text']}\"")
        if e['category']:
            print(f"    Category: {e['category']}")
        print()

    counts = get_escalation_counts_by_patient(days=30)
    repeat_patients = {p: c for p, c in counts.items() if c >= 2}
    if repeat_patients:
        print("=== Patients with repeated escalations (last 30 days) ===")
        for p, c in sorted(repeat_patients.items(), key=lambda x: -x[1]):
            print(f"  {p}: {c} escalations")

if __name__ == "__main__":
    main()