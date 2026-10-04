# src/rag/eval_reliability.py
from rag.qa_agent import answer_question
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

TEST_QUESTIONS = [
    "وقتاش لازم ناخذ الدواء، قبل ولا بعد الأكلة؟",
    "شنوة الأعراض الجانبية متاع هذا الدواء؟",
    "شنوة أحسن ماكلة في العيد الكبير",
    "كيفاش نجدد الوصفة متاع الدواء",
    "وقتاش لازم نتصل بالطبيب",
] * 2  # reduced from 4 repeats to 2, lighter run

results = [answer_question(q) for q in TEST_QUESTIONS]
fallback_count = sum(1 for r in results if "مشكل تقني" in r["answer"])
print(f"\nFallback triggered: {fallback_count}/{len(results)} ({fallback_count/len(results)*100:.0f}%)")