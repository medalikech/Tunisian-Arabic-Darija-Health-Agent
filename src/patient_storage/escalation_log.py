# src/patient_storage/escalation_log.py
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone

from patient_storage.data_crypto import (
    decrypt_text,
    encrypt_text,
    ensure_database_key,
)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "conversations.db"

def init_escalation_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_at TEXT,
                reviewer_note TEXT
            )
        """)
        conn.commit()

def log_escalation(patient_id: str, raw_text: str, severity: str, category: str = None):
    """severity: 'emergency' (rule-based red flag) or 'soft' (classifier soft-escalation)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        ensure_database_key(conn)
        conn.execute(
            "INSERT INTO escalations (timestamp, patient_id, raw_text, severity, category, status) "
            "VALUES (?, ?, ?, ?, ?, 'pending')",
            (
                datetime.now(timezone.utc).isoformat(),
                patient_id,
                encrypt_text(raw_text),
                severity,
                category,
            ),
        )
        conn.commit()

def get_pending_escalations() -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM escalations WHERE status = 'pending' ORDER BY timestamp DESC"
        ).fetchall()
    escalations = [dict(row) for row in rows]
    for escalation in escalations:
        escalation["raw_text"] = decrypt_text(escalation["raw_text"])
        escalation["reviewer_note"] = decrypt_text(escalation["reviewer_note"])
    return escalations

def resolve_escalation(escalation_id: int, note: str = ""):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        ensure_database_key(conn)
        conn.execute(
            "UPDATE escalations SET status = 'resolved', reviewed_at = ?, reviewer_note = ? WHERE id = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                encrypt_text(note),
                escalation_id,
            ),
        )
        conn.commit()

def get_escalation_counts_by_patient(days: int = 30) -> dict:
    """Useful for spotting patients with repeated escalations — matches the
    'proactive monitoring' goal from the original project brief."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT patient_id, COUNT(*) FROM escalations "
            "WHERE timestamp >= datetime('now', ?) GROUP BY patient_id",
            (f"-{days} days",),
        ).fetchall()
    return dict(rows)