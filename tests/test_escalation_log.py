import sqlite3
import unittest
import base64
import os
from unittest.mock import Mock, patch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from patient_storage import escalation_log


class EscalationLogConnectionTests(unittest.TestCase):
    def setUp(self):
        self.key_patcher = patch.dict(
            os.environ,
            {
                "DARJA_DATA_ENCRYPTION_KEY": base64.urlsafe_b64encode(
                    b"test-only-encryption-key-32bytes"
                ).decode("ascii")
            },
        )
        self.key_patcher.start()
        self.addCleanup(self.key_patcher.stop)
        self.key_check_patcher = patch.object(
            escalation_log,
            "ensure_database_key",
        )
        self.key_check_patcher.start()
        self.addCleanup(self.key_check_patcher.stop)

    def test_database_connections_close_after_successful_operations(self):
        operations = [
            escalation_log.init_escalation_db,
            lambda: escalation_log.log_escalation(
                "patient-test",
                "test emergency",
                severity="emergency",
                category="chest_pain",
            ),
            escalation_log.get_pending_escalations,
            lambda: escalation_log.resolve_escalation(1),
            escalation_log.get_escalation_counts_by_patient,
        ]

        for operation in operations:
            with self.subTest(operation=operation):
                connection = Mock()
                connection.execute.return_value.fetchall.return_value = []
                with patch.object(
                    escalation_log.sqlite3,
                    "connect",
                    return_value=connection,
                ):
                    operation()
                connection.close.assert_called_once_with()

    def test_database_connection_closes_when_insert_fails(self):
        connection = Mock()
        connection.execute.side_effect = sqlite3.OperationalError(
            "database unavailable"
        )

        with (
            patch.object(
                escalation_log.sqlite3,
                "connect",
                return_value=connection,
            ),
            self.assertRaises(sqlite3.OperationalError),
        ):
            escalation_log.log_escalation(
                "patient-test",
                "test emergency",
                severity="emergency",
                category="chest_pain",
            )

        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
