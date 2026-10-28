# src/storage/conversation_log.py
import json
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

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                route TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                matched_red_flags TEXT,
                response TEXT NOT NULL
            )
        """)
        conn.commit()

def log_interaction(patient_id: str, raw_text: str, result: dict):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        ensure_database_key(conn)
        conn.execute(
            """INSERT INTO messages
               (timestamp, patient_id, raw_text, route, intent, confidence, matched_red_flags, response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                patient_id,
                encrypt_text(raw_text),
                result.get("route"),
                result.get("intent"),
                result.get("confidence"),
                json.dumps(result.get("matched_categories")) if result.get("matched_categories") else None,
                encrypt_text(result.get("response")),
            ),
        )
        conn.commit()

def get_patient_history(patient_id: str, limit: int = 20) -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM messages WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
            (patient_id, limit),
        ).fetchall()
    history = [dict(row) for row in rows]
    for row in history:
        row["raw_text"] = decrypt_text(row["raw_text"])
        row["response"] = decrypt_text(row["response"])
    return history

def count_recent_red_flags(patient_id: str, days: int = 7) -> int:
    """Useful for the proactive-monitoring goal from the original project brief:
    flags a patient who's triggered the safety layer repeatedly in a short window."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM messages
               WHERE patient_id = ? AND route = 'emergency'
               AND timestamp >= datetime('now', ?)""",
            (patient_id, f"-{days} days"),
        ).fetchone()
    return row[0]
def get_recent_context(patient_id: str, n_turns: int = 3) -> list[dict]:
    """Returns the last n_turns (user message + response) for this patient,
    oldest first, for use as conversation context. Excludes the current
    in-progress message (call this before logging the new one)."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT raw_text, response, route FROM messages WHERE patient_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (patient_id, n_turns),
        ).fetchall()
    context = [dict(row) for row in reversed(rows)]
    for row in context:
        row["raw_text"] = decrypt_text(row["raw_text"])
        row["response"] = decrypt_text(row["response"])
    return context