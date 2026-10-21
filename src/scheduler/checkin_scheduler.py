# src/scheduler/checkin_scheduler.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.adherence_log import (
    init_adherence_db,
    get_patients_due_for_checkin,
    mark_checkin_sent,
)
from patient_storage.conversation_log import init_db, log_interaction

CHECKIN_MESSAGE = (
    "\u0623\u0647\u0644\u0627! \u0648\u0642\u062a\u0627\u0634 \u062e\u0630\u064a\u062a "
    "\u062f\u0648\u0627\u0621\u0643 \u0627\u0644\u064a\u0648\u0645\u061f \u0623\u062c\u0627\u0648\u0628\u0646\u064a "
    "\u0628\u00ab\u0646\u0639\u0645\u00bb \u0648\u0644\u0627 \u00ab\u0644\u0627\u00bb \u0648\u0646\u0635\u062c\u0644\u0647\u0627."
)

def run_checkins(dry_run: bool = False):
    init_db()
    init_adherence_db()

    due_patients = get_patients_due_for_checkin()
    print(f"Patients due for check-in: {len(due_patients)}")

    for patient_id in due_patients:
        print(f"\n  -> Sending check-in to: {patient_id}")
        print(f"     Message: {CHECKIN_MESSAGE}")

        if not dry_run:
            log_interaction(patient_id, "[SYSTEM: proactive check-in sent]", {
                "route": "proactive_checkin",
                "intent": None,
                "confidence": None,
                "response": CHECKIN_MESSAGE,
            })
            mark_checkin_sent(patient_id)

    print(f"\nDone. {len(due_patients)} check-in(s) {'would be ' if dry_run else ''}sent.")

if __name__ == "__main__":
    import sys as _sys
    dry = "--dry-run" in _sys.argv
    run_checkins(dry_run=dry)