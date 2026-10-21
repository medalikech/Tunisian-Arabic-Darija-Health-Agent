# src/scheduler/test_checkin_flow.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.adherence_log import register_patient_for_checkins, get_adherence_stats
from scheduler.checkin_scheduler import run_checkins
from agents.router import handle_message

patient = "checkin_flow_test"
register_patient_for_checkins(patient)

print("=== Running scheduler ===")
run_checkins()

print("\n=== Patient replies ===")
result = handle_message("نعم خذيت الدواء", patient_id=patient)
print(f"Route: {result['route']} | Response: {result['response']}")

print("\n=== Adherence stats ===")
print(get_adherence_stats(patient, days=7))