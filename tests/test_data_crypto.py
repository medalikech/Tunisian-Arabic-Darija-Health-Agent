import base64
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from patient_storage.data_crypto import (
    EncryptionConfigurationError,
    ENCRYPTED_PREFIX,
    EncryptionConfigurationError,
    decrypt_text,
    encrypt_text,
)
from patient_storage.migrate_encryption import migrate_database
from patient_storage import conversation_log, escalation_log


TEST_KEY = base64.urlsafe_b64encode(b"test-only-encryption-key-32bytes").decode(
    "ascii"
)


class DataCryptoTests(unittest.TestCase):
    def test_encrypted_value_round_trips_and_does_not_expose_plaintext(self):
        plaintext = "Sensitive test conversation"
        with patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}):
            encrypted = encrypt_text(plaintext)

            self.assertTrue(encrypted.startswith(ENCRYPTED_PREFIX))
            self.assertNotIn(plaintext, encrypted)
            self.assertEqual(decrypt_text(encrypted), plaintext)

    def test_plaintext_legacy_values_and_none_remain_readable(self):
        self.assertEqual(decrypt_text("legacy test text"), "legacy test text")
        self.assertIsNone(decrypt_text(None))
        self.assertIsNone(encrypt_text(None))

    def test_encrypting_requires_a_configured_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EncryptionConfigurationError):
                encrypt_text("Sensitive text")

    def test_wrong_key_fails_authentication(self):
        with patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}):
            encrypted = encrypt_text("Sensitive test conversation")

        other_key = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")
        with patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": other_key}):
            with self.assertRaisesRegex(ValueError, "could not be authenticated"):
                decrypt_text(encrypted)

    def test_migration_creates_backup_and_encrypts_existing_fields(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "conversations.db"
            backup_path = Path(temp_directory) / "before-encryption.db"
            with closing(sqlite3.connect(database_path)) as connection:
                connection.execute(
                    "CREATE TABLE messages (id INTEGER PRIMARY KEY, raw_text TEXT, response TEXT)"
                )
                connection.execute(
                    "INSERT INTO messages VALUES (1, 'patient text', 'assistant reply')"
                )
                connection.execute(
                    "CREATE TABLE escalations ("
                    "id INTEGER PRIMARY KEY, raw_text TEXT, reviewer_note TEXT)"
                )
                connection.execute(
                    "INSERT INTO escalations VALUES (1, 'emergency text', 'review note')"
                )
                connection.commit()

            with patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}):
                count = migrate_database(database_path, backup_path)
                with closing(sqlite3.connect(database_path)) as connection:
                    current_messages = connection.execute(
                        "SELECT raw_text, response FROM messages"
                    ).fetchone()
                    current_escalation = connection.execute(
                        "SELECT raw_text, reviewer_note FROM escalations"
                    ).fetchone()
                with closing(sqlite3.connect(backup_path)) as connection:
                    backup_messages = connection.execute(
                        "SELECT raw_text, response FROM messages"
                    ).fetchone()
                decrypted_note = decrypt_text(current_escalation[1])

            self.assertEqual(count, 4)
            self.assertTrue(all(value.startswith(ENCRYPTED_PREFIX) for value in current_messages))
            self.assertTrue(all(value.startswith(ENCRYPTED_PREFIX) for value in current_escalation))
            self.assertEqual(decrypted_note, "review note")
            self.assertEqual(
                backup_messages,
                ("patient text", "assistant reply"),
            )

    def test_migration_refuses_to_overwrite_a_backup(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "conversations.db"
            backup_path = Path(temp_directory) / "backup.db"
            connection = sqlite3.connect(database_path)
            connection.close()
            backup_path.write_text("preserve existing backup", encoding="utf-8")

            with (
                patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}),
                self.assertRaises(FileExistsError),
            ):
                migrate_database(database_path, backup_path)

            self.assertEqual(
                backup_path.read_text(encoding="utf-8"),
                "preserve existing backup",
            )

    def test_conversation_storage_encrypts_database_values_and_decrypts_reads(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "conversations.db"
            with (
                patch.object(conversation_log, "DB_PATH", database_path),
                patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}),
            ):
                conversation_log.init_db()
                conversation_log.log_interaction(
                    "patient-test",
                    "private message",
                    {"route": "fixed_response", "response": "private answer"},
                )
                history = conversation_log.get_patient_history("patient-test")
                context = conversation_log.get_recent_context("patient-test")

            with closing(sqlite3.connect(database_path)) as connection:
                stored_message, stored_response = connection.execute(
                    "SELECT raw_text, response FROM messages"
                ).fetchone()

            self.assertTrue(stored_message.startswith(ENCRYPTED_PREFIX))
            self.assertTrue(stored_response.startswith(ENCRYPTED_PREFIX))
            self.assertEqual(history[0]["raw_text"], "private message")
            self.assertEqual(history[0]["response"], "private answer")
            self.assertEqual(context[0]["raw_text"], "private message")
            self.assertEqual(context[0]["response"], "private answer")

    def test_database_rejects_a_different_key_after_initialization(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "conversations.db"
            with patch.object(conversation_log, "DB_PATH", database_path):
                conversation_log.init_db()
                with patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}):
                    conversation_log.log_interaction(
                        "patient-test",
                        "first message",
                        {"route": "fixed_response", "response": "first reply"},
                    )
                wrong_key = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")
                with (
                    patch.dict(
                        os.environ,
                        {"DARJA_DATA_ENCRYPTION_KEY": wrong_key},
                    ),
                    self.assertRaises(EncryptionConfigurationError),
                ):
                    conversation_log.log_interaction(
                        "patient-test",
                        "second message",
                        {"route": "fixed_response", "response": "second reply"},
                    )

            with closing(sqlite3.connect(database_path)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0]
            self.assertEqual(count, 1)

    def test_escalation_storage_encrypts_message_and_reviewer_note(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "conversations.db"
            with (
                patch.object(escalation_log, "DB_PATH", database_path),
                patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}),
            ):
                escalation_log.init_escalation_db()
                escalation_log.log_escalation(
                    "patient-test",
                    "private emergency details",
                    severity="emergency",
                    category="chest_pain",
                )
                escalation_log.resolve_escalation(1, "private review note")
                pending = escalation_log.get_pending_escalations()

            with closing(sqlite3.connect(database_path)) as connection:
                stored_message, stored_note = connection.execute(
                    "SELECT raw_text, reviewer_note FROM escalations"
                ).fetchone()

            self.assertTrue(stored_message.startswith(ENCRYPTED_PREFIX))
            self.assertTrue(stored_note.startswith(ENCRYPTED_PREFIX))
            self.assertEqual(pending, [])
            with (
                patch.object(escalation_log, "DB_PATH", database_path),
                patch.dict(os.environ, {"DARJA_DATA_ENCRYPTION_KEY": TEST_KEY}),
            ):
                escalation_log.log_escalation(
                    "patient-test",
                    "another emergency",
                    severity="emergency",
                    category="breathing",
                )
                pending = escalation_log.get_pending_escalations()
            self.assertEqual(pending[0]["raw_text"], "another emergency")
            self.assertIsNone(pending[0]["reviewer_note"])


if __name__ == "__main__":
    unittest.main()
