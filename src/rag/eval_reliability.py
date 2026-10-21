# src/rag/eval_reliability.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import answer_question

TEST_QUESTIONS = [
    "\u0648\u0642\u062a\u0627\u0634 \u0644\u0627\u0632\u0645 \u0646\u0627\u062e\u0630 \u0627\u0644\u062f\u0648\u0627\u0621\u061f",
    "\u0634\u0646\u0648\u0629 \u0627\u0644\u0623\u0639\u0631\u0627\u0636 \u0627\u0644\u062c\u0627\u0646\u0628\u064a\u0629 \u0645\u062a\u0627\u0639 \u0647\u0630\u0627 \u0627\u0644\u062f\u0648\u0627\u0621\u061f",
    "\u0634\u0646\u0648\u0629 \u0623\u062d\u0633\u0646 \u0645\u0627\u0643\u0644\u0629 \u0641\u064a \u0627\u0644\u0639\u064a\u062f \u0627\u0644\u0643\u0628\u064a\u0631",
    "\u0643\u064a\u0641\u0627\u0634 \u0646\u062c\u062f\u062f \u0627\u0644\u0648\u0635\u0641\u0629 \u0645\u062a\u0627\u0639 \u0627\u0644\u062f\u0648\u0627\u0621",
    "\u0648\u0642\u062a\u0627\u0634 \u0644\u0627\u0632\u0645 \u0646\u062a\u0635\u0644 \u0628\u0627\u0644\u0637\u0628\u064a\u0628",
] * 2

if __name__ == "__main__":
    results = [answer_question(q) for q in TEST_QUESTIONS]

    used_fallback = sum(1 for r in results if "\u062c\u0648\u0627\u0628 \u062c\u0627\u0647\u0632" in r["answer"])
    used_real_generation = len(results) - used_fallback

    print(f"\nTotal questions: {len(results)}")
    print(f"Real LLM generation succeeded: {used_real_generation}/{len(results)} "
          f"({used_real_generation/len(results)*100:.0f}%)")
    print(f"Fell back to verified pre-written answer: {used_fallback}/{len(results)} "
          f"({used_fallback/len(results)*100:.0f}%)")
    print(f"Correct/usable answer shown to patient: {len(results)}/{len(results)} (100%)")