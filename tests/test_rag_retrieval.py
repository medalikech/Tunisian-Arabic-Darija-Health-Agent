import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import (
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    is_retrieval_relevant,
    retrieve,
    retrieval_terms,
)


class FakeModel:
    def encode(self, *_args, **_kwargs):
        class Embedding:
            def tolist(self):
                return [[0.1, 0.2]]

        return Embedding()


class FakeCollection:
    def __init__(self, documents, distances, metadatas):
        self.documents = documents
        self.distances = distances
        self.metadatas = metadatas
        self.requested_count = None

    def query(self, **kwargs):
        n_results = kwargs["n_results"]
        self.requested_count = n_results
        count = min(n_results, len(self.documents))
        return {
            "documents": [self.documents[:count]],
            "distances": [self.distances[:count]],
            "metadatas": [self.metadatas[:count]],
        }


class RagRetrievalTests(unittest.TestCase):
    def test_arabic_metformin_and_food_terms_match_french_source(self):
        terms = retrieval_terms("وقتاش ناخذ المتفورمين مع الماكلة ولا قبلها؟")

        self.assertIn("metformin", terms)
        self.assertIn("repas", terms)

    def test_negation_is_not_removed_from_lexical_terms(self):
        terms = retrieval_terms("ما تضاعفش الجرعة")

        self.assertIn("ما", terms)

    def test_arabic_punctuation_and_attached_prefixes_are_normalized(self):
        terms = retrieval_terms(
            "\u0628\u0627\u0644\u0645\u062a\u0641\u0648\u0631\u0645\u064a\u0646"
            "\u060c \u0648\u0627\u0644\u0633\u0643\u0631\u061f"
        )

        self.assertIn("metformin", terms)
        self.assertIn("glycemie", terms)

    def test_arabizi_terms_normalize_to_known_french_concepts(self):
        terms = retrieval_terms(
            "wa9tech ne5ou metformine ma3a el mekla wala 9balha?"
        )

        self.assertTrue(
            {"metformin", "repas"}.issubset(terms)
        )
        self.assertNotIn("prendre", terms)

    def test_retrieval_gate_accepts_arabizi_with_semantic_and_lexical_support(self):
        chunks = [
            (
                "Mesurer la glycemie avec un glucometre.",
                0.5,
                {"darija_fallback": ""},
            )
        ]

        self.assertTrue(
            is_retrieval_relevant(
                "kifech na9is soukkar bel glucometre?",
                chunks,
            )
        )

    def test_retrieval_gate_rejects_semantically_similar_but_lexically_unrelated(self):
        chunks = [("Conseils sur la glycemie.", 0.9, {"darija_fallback": ""})]

        self.assertFalse(
            is_retrieval_relevant("capital tounes", chunks)
        )

    def test_retrieval_gate_requires_semantic_and_lexical_support_in_same_chunk(self):
        chunks = [
            ("Conseils sur le diabete.", 0.9, {"darija_fallback": ""}),
            (
                "La metformine se prend pendant les repas.",
                0.1,
                {"darija_fallback": ""},
            ),
        ]

        self.assertFalse(
            is_retrieval_relevant("metformine repas", chunks)
        )

    def test_retrieval_gate_rejects_low_semantic_score_and_empty_results(self):
        chunks = [
            (
                "Mesurer la glycemie avec un glucometre.",
                0.1,
                {"darija_fallback": ""},
            )
        ]

        self.assertFalse(
            is_retrieval_relevant(
                "kifech na9is soukkar bel glucometre?",
                chunks,
            )
        )
        self.assertFalse(
            is_retrieval_relevant(
                "kifech na9is soukkar bel glucometre?",
                [],
            )
        )

    def test_lexical_match_can_rerank_relevant_chunk(self):
        collection = FakeCollection(
            documents=[
                "Unrelated travel advice",
                "Metformine se prend pendant les repas",
            ],
            distances=[0.31, 0.6],
            metadatas=[
                {"source": "travel.txt", "darija_fallback": "معلومات عن السفر"},
                {
                    "source": "metformin.txt",
                    "darija_fallback": "وقتاش تاخذ الدواء مع الأكلة",
                },
            ],
        )

        results = retrieve(
            "وقتاش ناخذ المتفورمين مع الماكلة؟",
            FakeModel(),
            collection,
            k=1,
        )

        self.assertEqual(collection.requested_count, RETRIEVAL_CANDIDATE_MULTIPLIER)
        self.assertEqual(results[0][2]["source"], "metformin.txt")
        self.assertAlmostEqual(results[0][1], 0.4)

    def test_results_include_at_most_one_chunk_per_source(self):
        collection = FakeCollection(
            documents=[
                "First relevant chunk",
                "Second chunk from same source",
                "Chunk from another source",
            ],
            distances=[0.2, 0.3, 0.4],
            metadatas=[
                {"source": "first.txt", "darija_fallback": ""},
                {"source": "first.txt", "darija_fallback": ""},
                {"source": "second.txt", "darija_fallback": ""},
            ],
        )

        results = retrieve("unrelated query", FakeModel(), collection, k=3)

        self.assertEqual(
            [metadata["source"] for _, _, metadata in results],
            ["first.txt", "second.txt"],
        )
        self.assertEqual(
            [document for document, _, _ in results],
            ["First relevant chunk", "Chunk from another source"],
        )


if __name__ == "__main__":
    unittest.main()
