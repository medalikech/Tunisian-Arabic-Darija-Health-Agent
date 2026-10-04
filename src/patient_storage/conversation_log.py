# src/storage/conversation_log.py
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "conversations.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()

def log_interaction(patient_id: str, raw_text: str, result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO messages
           (timestamp, patient_id, raw_text, route, intent, confidence, matched_red_flags, response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            patient_id,
            raw_text,
            result.get("route"),
            result.get("intent"),
            result.get("confidence"),
            json.dumps(result.get("matched_categories")) if result.get("matched_categories") else None,
            result.get("response"),
        ),
    )
    conn.commit()
    conn.close()

def get_patient_history(patient_id: str, limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM messages WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
        (patient_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def count_recent_red_flags(patient_id: str, days: int = 7) -> int:
    """Useful for the proactive-monitoring goal from the original project brief:
    flags a patient who's triggered the safety layer repeatedly in a short window."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """SELECT COUNT(*) FROM messages
           WHERE patient_id = ? AND route = 'emergency'
           AND timestamp >= datetime('now', ?)""",
        (patient_id, f"-{days} days"),
    ).fetchone()
    conn.close()
    return row[0]