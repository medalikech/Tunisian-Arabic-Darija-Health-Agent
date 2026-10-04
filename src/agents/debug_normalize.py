# src/agents/debug_normalize.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from nlp.normalize import normalize

test_messages = [
    "وقتاش لازم ناخذ الدواء",
    "نحس بالضيق من هذا المرض",
    "سلام كيفاش الحال",
]

for msg in test_messages:
    print(f"IN:  {msg!r}")
    print(f"OUT: {normalize(msg)!r}\n")