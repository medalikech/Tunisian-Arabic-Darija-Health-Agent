# src/agents/check_log.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.conversation_log import get_patient_history

for row in get_patient_history("test_patient"):
    print(f"[{row['route']}] {row['raw_text'][:40]} -> {row['response'][:50]}")