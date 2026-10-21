# src/rag/sample_outputs.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import answer_question

questions = [
    "\u0648\u0642\u062a\u0627\u0634 \u0644\u0627\u0632\u0645 \u0646\u0627\u062e\u0630 \u0627\u0644\u062f\u0648\u0627\u0621\u061f",
    "\u0634\u0646\u0648\u0629 \u0627\u0644\u0623\u0639\u0631\u0627\u0636 \u0627\u0644\u062c\u0627\u0646\u0628\u064a\u0629 \u0645\u062a\u0627\u0639 \u0647\u0630\u0627 \u0627\u0644\u062f\u0648\u0627\u0621\u061f",
]

for q in questions:
    result = answer_question(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")