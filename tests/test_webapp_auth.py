import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from webapp import auth


class WebAppAuthTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_patcher = patch.object(
            auth,
            "AUTH_DB_PATH",
            Path(self.temp_dir.name) / "auth.db",
        )
        self.db_patcher.start()
        self.addCleanup(self.db_patcher.stop)
        auth.initialize_auth_db()

    def test_password_is_stored_as_a_hash(self):
        user = auth.create_user(
            "patient",
            "correct-horse-battery",
            "patient",
            "patient-123",
        )

        with auth._connection() as conn:
            stored_hash = conn.execute(
                "SELECT password_hash FROM users WHERE id = ?",
                (user["id"],),
            ).fetchone()[0]

        self.assertNotEqual(stored_hash, "correct-horse-battery")
        self.assertTrue(stored_hash.startswith("scrypt:"))
        self.assertIsNotNone(auth.authenticate_user("patient", "correct-horse-battery"))
        self.assertIsNone(auth.authenticate_user("patient", "incorrect-password"))

    def test_short_password_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 12 and 1024 characters"):
            auth.create_user("patient", "too-short", "patient", "patient-123")

    def test_repeated_login_failures_temporarily_lock_account(self):
        auth.create_user(
            "patient",
            "correct-horse-battery",
            "patient",
            "patient-123",
        )
        now = 1_000_000

        with patch.object(auth.time, "time", return_value=now):
            for _ in range(auth.MAX_LOGIN_ATTEMPTS):
                self.assertIsNone(auth.authenticate_user("patient", "wrong-password"))
            self.assertIsNone(
                auth.authenticate_user("patient", "correct-horse-battery")
            )

        with patch.object(
            auth.time,
            "time",
            return_value=now + auth.LOGIN_LOCKOUT_SECONDS + 1,
        ):
            self.assertIsNotNone(
                auth.authenticate_user("patient", "correct-horse-battery")
            )

    def test_successful_login_clears_failed_attempts(self):
        auth.create_user(
            "patient",
            "correct-horse-battery",
            "patient",
            "patient-123",
        )

        for _ in range(auth.MAX_LOGIN_ATTEMPTS - 1):
            self.assertIsNone(auth.authenticate_user("patient", "wrong-password"))
        self.assertIsNotNone(
            auth.authenticate_user("patient", "correct-horse-battery")
        )
        self.assertIsNone(auth.authenticate_user("patient", "wrong-password"))

        with auth._connection() as conn:
            failure = conn.execute(
                "SELECT failed_attempts, locked_until FROM login_failures"
            ).fetchone()

        self.assertEqual(failure, (1, None))

    def test_patient_access_is_bound_to_their_own_id(self):
        patient = auth.create_user(
            "patient",
            "correct-horse-battery",
            "patient",
            "patient-123",
        )

        self.assertTrue(auth.can_access_patient(patient, "patient-123"))
        self.assertFalse(auth.can_access_patient(patient, "patient-456"))


if __name__ == "__main__":
    unittest.main()
