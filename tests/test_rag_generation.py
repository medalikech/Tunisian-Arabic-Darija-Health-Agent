import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag import qa_agent
from rag.qa_agent import generate_answer


class RagGenerationTests(unittest.TestCase):
    @patch("rag.qa_agent.generate_answer")
    @patch("rag.qa_agent.retrieve")
    @patch("rag.qa_agent.get_collection")
    @patch("rag.qa_agent.embedder")
    @patch("rag.qa_agent.is_retrieval_relevant", return_value=True)
    def test_answer_question_passes_french_and_darija_evidence(
        self,
        is_retrieval_relevant,
        embedder,
        get_collection,
        retrieve,
        generate,
    ):
        fallback = (
            "\u0644\u0627\u0632\u0645 \u062a\u0627\u062e\u0630 "
            "\u0627\u0644\u062f\u0648\u0627\u0621 \u0645\u0639 "
            "\u0627\u0644\u0623\u0643\u0644\u0629."
        )
        retrieve.return_value = [
            (
                "French source about taking medicine with meals.",
                0.8,
                {"source": "medicine.txt", "darija_fallback": fallback},
            )
        ]
        generate.return_value = "Generated response"

        result = qa_agent.answer_question("A question")

        self.assertEqual(result["answer"], "Generated response")
        self.assertEqual(result["source_files"], ["medicine.txt"])
        context_chunks = generate.call_args.args[1]
        self.assertIn("French source about taking medicine", context_chunks[0])
        self.assertIn(fallback, context_chunks[0])
        self.assertEqual(
            generate.call_args.kwargs["grounding_chunks"],
            [fallback],
        )

    @patch("rag.qa_agent.generate_answer")
    @patch("rag.qa_agent.retrieve", return_value=[])
    @patch("rag.qa_agent.get_collection")
    @patch("rag.qa_agent.embedder")
    def test_retrieval_refusal_has_no_sources_and_skips_generation(
        self,
        embedder,
        get_collection,
        retrieve,
        generate,
    ):
        result = qa_agent.answer_question("An unsupported question")

        self.assertFalse(result["answered"])
        self.assertEqual(result["source_files"], [])
        generate.assert_not_called()

    @patch("rag.qa_agent.requests.post")
    def test_accepts_darija_answer_grounded_by_fallback(self, post):
        answer = (
            "\u0644\u0627\u0632\u0645 \u062a\u0627\u062e\u0630 "
            "\u0627\u0644\u062f\u0648\u0627\u0621 \u0645\u0639 "
            "\u0627\u0644\u0623\u0643\u0644\u0629 \u0628\u0627\u0634 "
            "\u062a\u0642\u0644 \u0627\u0644\u0645\u0634\u0627\u0643\u0644 "
            "\u0641\u064a \u0627\u0644\u0645\u0639\u062f\u0629."
        )
        post.return_value.json.return_value = {"response": answer}

        result = generate_answer(
            "When should I take this medicine?",
            ["French source about taking medicine with meals."],
            grounding_chunks=[
                "\u0644\u0627\u0632\u0645 \u062a\u0627\u062e\u0630 "
                "\u0627\u0644\u062f\u0648\u0627\u0621 \u0645\u0639 "
                "\u0627\u0644\u0623\u0643\u0644\u0629 \u0628\u0627\u0634 "
                "\u062a\u0642\u0644 \u0627\u0644\u0645\u0634\u0627\u0643\u0644 "
                "\u0641\u064a \u0627\u0644\u0645\u0639\u062f\u0629."
            ],
        )

        self.assertEqual(result, answer)
        post.assert_called_once()
        prompt = post.call_args.kwargs["json"]["prompt"]
        self.assertIn("French source about taking medicine", prompt)

    @patch("rag.qa_agent.requests.post")
    def test_rejects_answer_without_overlap_with_any_grounding_chunk(self, post):
        answer = (
            "\u064a\u0646\u062c\u0645 \u062a\u0645\u0634\u064a "
            "\u0644\u0644\u0635\u064a\u062f\u0644\u064a\u0629 "
            "\u0648\u062a\u0633\u0623\u0644\u0647\u0645 \u0639\u0644\u0649 "
            "\u0645\u0648\u0636\u0648\u0639 \u0622\u062e\u0631."
        )
        post.return_value.json.return_value = {"response": answer}

        result = generate_answer(
            "A question",
            ["French medical source."],
            retries=1,
            grounding_chunks=[
                "\u0627\u0644\u062f\u0648\u0627\u0621 \u064a\u062a\u062e\u0630 "
                "\u0645\u0639 \u0627\u0644\u0623\u0643\u0644\u0629 "
                "\u0644\u062a\u0642\u0644\u064a\u0644 \u0627\u0644\u063a\u062b\u064a\u0627\u0646."
            ],
        )

        self.assertIn(
            "\u0645\u0634\u0643\u0644 \u062a\u0642\u0646\u064a",
            result,
        )
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
