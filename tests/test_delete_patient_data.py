import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from patient_storage.delete_patient_data import delete_patient_data


class DeletePatientDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        root = Path(self.temp_directory.name)
        self.conversations_db = root / "conversations.db"
        self.auth_db = root / "auth.db"
        self._create_conversation_database()
        self._create_auth_database()

    def _create_conversation_database(self):
        with closing(sqlite3.connect(self.conversations_db)) as connection:
            connection.executescript("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    raw_text TEXT NOT NULL
                );
                CREATE TABLE dose_events (
                    id INTEGER PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    event_type TEXT NOT NULL
                );
                CREATE TABLE checkin_schedule (
                    id INTEGER PRIMARY KEY,
                    patient_id TEXT NOT NULL UNIQUE
                );
                CREATE TABLE escalations (
                    id INTEGER PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    raw_text TEXT NOT NULL
                );
                INSERT INTO messages VALUES
                    (1, 'patient-delete', 'message'),
                    (2, 'patient-keep', 'other message');
                INSERT INTO dose_events VALUES
                    (1, 'patient-delete', 'taken'),
                    (2, 'patient-keep', 'missed');
                INSERT INTO checkin_schedule VALUES
                    (1, 'patient-delete'),
                    (2, 'patient-keep');
                INSERT INTO escalations VALUES
                    (1, 'patient-delete', 'retained escalation'),
                    (2, 'patient-keep', 'other escalation');
            """)
            connection.commit()

    def _create_auth_database(self):
        with closing(sqlite3.connect(self.auth_db)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    patient_id TEXT
                );
                CREATE TABLE login_failures (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    failed_attempts INTEGER NOT NULL
                );
                CREATE TABLE staff_patient_access (
                    staff_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    patient_id TEXT NOT NULL
                );
                INSERT INTO users VALUES
                    (1, 'patient-delete', 'patient', 'patient-delete'),
                    (2, 'patient-keep', 'patient', 'patient-keep'),
                    (3, 'staff', 'staff', NULL);
                INSERT INTO login_failures VALUES (1, 2);
                INSERT INTO staff_patient_access VALUES
                    (3, 'patient-delete'),
                    (3, 'patient-keep');
            """)
            connection.commit()

    def test_deletes_patient_data_and_account_but_retains_escalation_audit(self):
        counts = delete_patient_data(
            "patient-delete",
            "patient-delete",
            self.conversations_db,
            self.auth_db,
        )

        with closing(sqlite3.connect(self.conversations_db)) as connection:
            messages = connection.execute(
                "SELECT patient_id FROM messages"
            ).fetchall()
            dose_events = connection.execute(
                "SELECT patient_id FROM dose_events"
            ).fetchall()
            schedules = connection.execute(
                "SELECT patient_id FROM checkin_schedule"
            ).fetchall()
            escalations = connection.execute(
                "SELECT patient_id, raw_text FROM escalations"
            ).fetchall()

        with closing(sqlite3.connect(self.auth_db)) as connection:
            users = connection.execute(
                "SELECT patient_id FROM users ORDER BY id"
            ).fetchall()
            access = connection.execute(
                "SELECT patient_id FROM staff_patient_access ORDER BY patient_id"
            ).fetchall()
            login_failures = connection.execute(
                "SELECT user_id FROM login_failures"
            ).fetchall()

        self.assertEqual(counts["messages"], 1)
        self.assertEqual(counts["dose_events"], 1)
        self.assertEqual(counts["checkin_schedule"], 1)
        self.assertEqual(counts["staff_patient_access"], 1)
        self.assertEqual(counts["escalations_retained"], 1)
        self.assertEqual(counts["patient_accounts"], 1)
        self.assertEqual(messages, [("patient-keep",)])
        self.assertEqual(dose_events, [("patient-keep",)])
        self.assertEqual(schedules, [("patient-keep",)])
        self.assertEqual(
            escalations,
            [
                ("patient-delete", "retained escalation"),
                ("patient-keep", "other escalation"),
            ],
        )
        self.assertEqual(users, [("patient-keep",), (None,)])
        self.assertEqual(access, [("patient-keep",)])
        self.assertEqual(login_failures, [])

    def test_requires_exact_confirmation_before_touching_databases(self):
        with self.assertRaisesRegex(ValueError, "exactly match"):
            delete_patient_data(
                "patient-delete",
                "patient-keep",
                self.conversations_db,
                self.auth_db,
            )

        with closing(sqlite3.connect(self.conversations_db)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                2,
            )

    def test_refuses_wal_mode_before_deleting_any_records(self):
        with closing(sqlite3.connect(self.auth_db)) as connection:
            connection.execute("PRAGMA journal_mode = WAL")

        with self.assertRaisesRegex(RuntimeError, "rollback journal"):
            delete_patient_data(
                "patient-delete",
                "patient-delete",
                self.conversations_db,
                self.auth_db,
            )

        with closing(sqlite3.connect(self.conversations_db)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                2,
            )

    def test_rolls_back_conversation_deletion_if_account_deletion_fails(self):
        with closing(sqlite3.connect(self.auth_db)) as connection:
            connection.execute("""
                CREATE TRIGGER prevent_patient_delete
                BEFORE DELETE ON users
                WHEN OLD.patient_id = 'patient-delete'
                BEGIN
                    SELECT RAISE(ABORT, 'account deletion blocked');
                END
            """)

        with self.assertRaisesRegex(sqlite3.IntegrityError, "account deletion blocked"):
            delete_patient_data(
                "patient-delete",
                "patient-delete",
                self.conversations_db,
                self.auth_db,
            )

        with closing(sqlite3.connect(self.conversations_db)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE patient_id = 'patient-delete'"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM dose_events WHERE patient_id = 'patient-delete'"
                ).fetchone()[0],
                1,
            )


if __name__ == "__main__":
    unittest.main()
