# src/rag/debug_string_match.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import generate_answer

failure_message = "\u0633\u0645\u062d\u0644\u064a\u060c \u0635\u0627\u0631 \u0645\u0634\u0643\u0644 \u062a\u0642\u0646\u064a \u0648\u0645\u0627 \u0642\u062f\u0631\u062a\u0634 \u0646\u0631\u062a\u0628 \u062c\u0648\u0627\u0628 \u0628\u0627\u0647\u064a. \u0639\u0627\u0648\u062f \u0627\u0644\u0633\u0624\u0627\u0644 \u0645\u0646 \u0628\u0639\u062f \u0648\u0644\u0627 \u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u0627\u0644\u0635\u064a\u062f\u0644\u064a\u0629."

check_string = "\u0645\u0634\u0643\u0644 \u062a\u0642\u0646\u064a"

print("Failure message:", failure_message)
print("Check string:", check_string)
print("Match?", check_string in failure_message)