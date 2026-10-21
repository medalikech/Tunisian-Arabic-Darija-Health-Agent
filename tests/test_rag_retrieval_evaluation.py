import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.eval_retrieval import (
    evaluate_relevance_cases,
    evaluate_retrieval_cases,
)


class RagRetrievalEvaluationTests(unittest.TestCase):
    def test_reports_recall_at_k_and_mean_reciprocal_rank(self):
        cases = [
            {
                "question": "Question one",
                "expected_sources": ["a.txt"],
                "language": "Arabic",
            },
            {
                "question": "Question two",
                "expected_sources": ["a.txt", "b.txt"],
                "language": "Arabizi/mixed",
            },
            {"question": "Question three", "expected_sources": ["c.txt"]},
        ]
        retrieved = {
            "Question one": [
                ("doc-b", 0.9, {"source": "b.txt"}),
                ("doc-a", 0.8, {"source": "a.txt"}),
            ],
            "Question two": [
                ("doc-b", 0.9, {"source": "b.txt"}),
                ("doc-x", 0.8, {"source": "x.txt"}),
            ],
            "Question three": [
                ("doc-x", 0.9, {"source": "x.txt"}),
            ],
        }

        results = evaluate_retrieval_cases(
            cases,
            lambda question, k: retrieved[question][:k],
            k=2,
        )

        self.assertEqual(results["case_count"], 3)
        self.assertAlmostEqual(results["recall_at_k"], 0.5)
        self.assertAlmostEqual(results["mrr"], 0.5)
        self.assertEqual(results["cases"][0]["reciprocal_rank"], 0.5)
        self.assertEqual(results["cases"][1]["recall_at_k"], 0.5)
        self.assertEqual(results["cases"][2]["reciprocal_rank"], 0)
        self.assertEqual(
            results["by_language"]["Arabic"],
            {"case_count": 2, "recall_at_k": 0.5, "mrr": 0.25},
        )
        self.assertEqual(
            results["by_language"]["Arabizi/mixed"],
            {"case_count": 1, "recall_at_k": 0.5, "mrr": 1.0},
        )

    def test_empty_case_list_returns_zero_metrics(self):
        results = evaluate_retrieval_cases([], lambda *_args: [])

        self.assertEqual(results["case_count"], 0)
        self.assertEqual(results["recall_at_k"], 0)
        self.assertEqual(results["mrr"], 0)
        self.assertEqual(results["by_language"], {})
        self.assertEqual(results["cases"], [])

    def test_case_requires_at_least_one_expected_source(self):
        with self.assertRaisesRegex(ValueError, "expected source"):
            evaluate_retrieval_cases(
                [{"question": "Question", "expected_sources": []}],
                lambda *_args: [],
            )

    def test_reports_supported_acceptance_and_out_of_scope_rejection(self):
        cases = [
            {"question": "supported", "in_scope": True, "category": "supported"},
            {"question": "missed supported", "in_scope": True, "category": "supported"},
            {"question": "unrelated", "in_scope": False, "category": "unrelated"},
            {"question": "false accept", "in_scope": False, "category": "medical"},
        ]
        retrieved = {
            "supported": [("doc", 0.8, {})],
            "missed supported": [],
            "unrelated": [],
            "false accept": [("doc", 0.8, {})],
        }

        results = evaluate_relevance_cases(
            cases,
            lambda question, _k: retrieved[question],
            lambda _question, chunks: bool(chunks),
        )

        self.assertEqual(results["case_count"], 4)
        self.assertEqual(results["accuracy"], 0.5)
        self.assertEqual(results["supported_acceptance_rate"], 0.5)
        self.assertEqual(results["out_of_scope_rejection_rate"], 0.5)
        self.assertEqual(
            results["by_category"]["medical"],
            {"case_count": 1, "acceptance_rate": 1.0, "accuracy": 0.0},
        )
        self.assertEqual(
            [case["outcome"] for case in results["false_refusals"]],
            ["false_refusal"],
        )
        self.assertEqual(
            [case["outcome"] for case in results["false_accepts"]],
            ["false_accept"],
        )

    def test_relevance_evaluation_rejects_missing_scope_label(self):
        with self.assertRaisesRegex(ValueError, "boolean in_scope"):
            evaluate_relevance_cases(
                [{"question": "Question"}],
                lambda *_args: [],
                lambda *_args: False,
            )

    def test_empty_relevance_cases_return_zero_rates(self):
        results = evaluate_relevance_cases(
            [],
            lambda *_args: [],
            lambda *_args: False,
        )

        self.assertEqual(results["case_count"], 0)
        self.assertEqual(results["accuracy"], 0)
        self.assertEqual(results["by_category"], {})
        self.assertEqual(results["supported_acceptance_rate"], 0)
        self.assertEqual(results["out_of_scope_rejection_rate"], 0)
        self.assertEqual(results["false_refusals"], [])
        self.assertEqual(results["false_accepts"], [])


if __name__ == "__main__":
    unittest.main()
