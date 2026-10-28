from contextlib import redirect_stdout
from io import StringIO
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from interface.cli import print_rag_debug


class CliRagDebugTests(unittest.TestCase):
    def test_prints_source_scores_and_gate_decision(self):
        output = StringIO()
        result = {
            "rag_sources": ["metformin.txt"],
            "rag_answered": True,
            "rag_retrieval_accepted": True,
            "rag_answer_status": "source_fallback",
            "rag_trace": [
                {
                    "source": "metformin.txt",
                    "semantic_score": 0.81234,
                    "lexical_matches": 3,
                    "lexical_coverage": 0.667,
                    "passes_relevance_gate": True,
                    "sent_to_generation": True,
                },
                {
                    "source": "unrelated.txt",
                    "semantic_score": 0.901,
                    "lexical_matches": 0,
                    "lexical_coverage": 0.0,
                    "passes_relevance_gate": False,
                    "sent_to_generation": False,
                },
            ],
        }

        with redirect_stdout(output):
            print_rag_debug(result)

        printed = output.getvalue()
        self.assertIn("RAG sources=metformin.txt", printed)
        self.assertIn("semantic=0.812", printed)
        self.assertIn("lexical_matches=3", printed)
        self.assertIn("passes_gate=True, sent_to_generation=True", printed)
        self.assertIn("source=unrelated.txt", printed)
        self.assertIn("passes_gate=False, sent_to_generation=False", printed)
        self.assertIn(
            "RAG answered=True, retrieval_accepted=True, "
            "answer_status=source_fallback",
            printed,
        )


if __name__ == "__main__":
    unittest.main()
