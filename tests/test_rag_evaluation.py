import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from eval.run_full_eval import summarize_rag_results


class RagEvaluationSummaryTests(unittest.TestCase):
    def test_distinguishes_all_rag_outcomes(self):
        results = [
            {"answered": True, "answer": "Generated answer."},
            {
                "answered": True,
                "answer": "Prepared answer.\n\n(جواب جاهز مبني على معلومات موثوقة)",
            },
            {"answered": True, "answer": "صار مشكل تقني"},
            {"answered": False, "answer": "لا توجد معلومات كافية"},
        ]

        self.assertEqual(
            summarize_rag_results(results),
            {
                "generated": 1,
                "prepared_fallback": 1,
                "technical_failure": 1,
                "retrieval_refusal": 1,
            },
        )

    def test_empty_results_have_zero_counts(self):
        self.assertEqual(
            summarize_rag_results([]),
            {
                "generated": 0,
                "prepared_fallback": 0,
                "technical_failure": 0,
                "retrieval_refusal": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
