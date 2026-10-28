import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from agents import router


class RouterRagSourceTests(unittest.TestCase):
    @patch("rag.qa_agent.answer_question")
    def test_rag_sources_are_forwarded_by_router(self, answer_question):
        answer_question.return_value = {
            "answer": "A grounded response",
            "best_score": 0.82,
            "source_files": ["metformin_basics.txt", "side_effects_detailed.txt"],
            "answered": True,
            "retrieval_accepted": True,
            "answer_status": "generated",
            "retrieval_trace": [
                {
                    "source": "metformin_basics.txt",
                    "semantic_score": 0.82,
                    "lexical_matches": 3,
                    "lexical_coverage": 0.75,
                    "passes_relevance_gate": True,
                    "sent_to_generation": True,
                }
            ],
        }
        with (
            patch.object(router, "scan_for_red_flags", return_value=[]),
            patch.object(router, "get_recent_context", return_value=[]),
            patch.object(
                router.CLASSIFIER,
                "predict",
                return_value=["ask_medication_info"],
            ),
            patch.object(
                router.CLASSIFIER,
                "predict_proba",
                return_value=[[0.9, 0.1]],
            ),
            patch.object(router, "log_interaction") as log_interaction,
        ):
            result = router.handle_message(
                "A medicine question",
                patient_id="rag-source-test",
            )

        self.assertEqual(result["route"], "rag")
        self.assertEqual(
            result["rag_sources"],
            ["metformin_basics.txt", "side_effects_detailed.txt"],
        )
        self.assertTrue(result["rag_answered"])
        self.assertTrue(result["rag_retrieval_accepted"])
        self.assertEqual(result["rag_answer_status"], "generated")
        self.assertEqual(
            result["rag_trace"],
            answer_question.return_value["retrieval_trace"],
        )
        log_interaction.assert_called_once()
        self.assertEqual(
            log_interaction.call_args.args[2]["rag_sources"],
            result["rag_sources"],
        )


if __name__ == "__main__":
    unittest.main()
