import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.build_index import index_knowledge_base


class FakeEmbeddings:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.encoded_documents = None

    def encode(self, documents, show_progress_bar):
        self.encoded_documents = documents
        return FakeEmbeddings([[index] for index, _ in enumerate(documents)])


class FakeCollection:
    def __init__(self):
        self.records = {}

    def get(self):
        return {"ids": list(self.records)}

    def upsert(self, ids, embeddings, documents, metadatas):
        for record_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            self.records[record_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)


class RagBuildIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.kb_dir = Path(self.temp_dir.name)
        self.collection = FakeCollection()

    def write_kb(self, contents):
        (self.kb_dir / "medicine.txt").write_text(contents, encoding="utf-8")

    def test_reindex_updates_chunks_and_removes_stale_ids(self):
        self.write_kb(
            "TITRE: First\nFrench content\nDARIJA:\nDarija content\n"
            "\nTITRE: Second\nMore French content\nDARIJA:\nMore Darija"
        )
        self.collection.records["legacy-chunk-id"] = {}

        first_count = index_knowledge_base(
            self.collection, FakeModel(), self.kb_dir
        )
        first_records = dict(self.collection.records)

        self.assertEqual(first_count, 2)
        self.assertEqual(len(first_records), 2)
        self.assertNotIn("legacy-chunk-id", self.collection.records)

        self.write_kb("TITRE: Updated\nUpdated French content\nDARIJA:\nUpdated")
        second_count = index_knowledge_base(
            self.collection, FakeModel(), self.kb_dir
        )

        self.assertEqual(second_count, 1)
        self.assertEqual(len(self.collection.records), 1)
        remaining = next(iter(self.collection.records.values()))
        self.assertEqual(remaining["document"], "TITRE: Updated\nUpdated French content")

    def test_rebuilding_same_content_reuses_existing_ids(self):
        self.write_kb("TITRE: Topic\nFrench content\nDARIJA:\nDarija content")

        model = FakeModel()
        index_knowledge_base(self.collection, model, self.kb_dir)
        first_ids = set(self.collection.records)
        index_knowledge_base(self.collection, FakeModel(), self.kb_dir)

        self.assertEqual(set(self.collection.records), first_ids)
        self.assertEqual(len(self.collection.records), 1)
        self.assertEqual(model.encoded_documents, ["Darija content"])
        record = next(iter(self.collection.records.values()))
        self.assertEqual(record["document"], "TITRE: Topic\nFrench content")
        self.assertEqual(record["metadata"]["darija_fallback"], "Darija content")

    def test_empty_knowledge_base_does_not_delete_existing_index(self):
        self.collection.records["existing-id"] = {"document": "existing"}

        with self.assertRaisesRegex(ValueError, "No knowledge-base chunks"):
            index_knowledge_base(self.collection, FakeModel(), self.kb_dir)

        self.assertIn("existing-id", self.collection.records)


if __name__ == "__main__":
    unittest.main()
