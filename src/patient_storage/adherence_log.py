# src/patient_storage/adherence_log.py
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, date

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "conversations.db"

def init_adherence_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dose_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkin_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL UNIQUE,
            active INTEGER NOT NULL DEFAULT 1,
            last_checkin TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_dose_event(patient_id: str, event_type: str, source: str = "conversation", event_date: str = None):
    """event_type: 'taken' or 'missed'. source: 'conversation' (inferred from chat)
    or 'checkin_reply' (patient answered a scheduled check-in)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO dose_events (patient_id, event_date, event_type, source, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            patient_id,
            event_date or date.today().isoformat(),
            event_type,
            source,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()

def get_adherence_stats(patient_id: str, days: int = 7) -> dict:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT event_type, COUNT(*) FROM dose_events "
        "WHERE patient_id = ? AND timestamp >= datetime('now', ?) "
        "GROUP BY event_type",
        (patient_id, f"-{days} days"),
    ).fetchall()
    conn.close()
    counts = {row[0]: row[1] for row in rows}
    taken = counts.get("taken", 0)
    missed = counts.get("missed", 0)
    total = taken + missed
    return {
        "taken": taken,
        "missed": missed,
        "total": total,
        "adherence_rate": round(taken / total, 2) if total else None,
    }

def register_patient_for_checkins(patient_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO checkin_schedule (patient_id, active, last_checkin) VALUES (?, 1, NULL)",
        (patient_id,),
    )
    conn.commit()
    conn.close()

def get_patients_due_for_checkin() -> list[str]:
    """Patients active in check-ins who haven't been checked in on today."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT patient_id FROM checkin_schedule WHERE active = 1 "
        "AND (last_checkin IS NULL OR date(last_checkin) < date('now'))"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

def mark_checkin_sent(patient_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE checkin_schedule SET last_checkin = ? WHERE patient_id = ?",
        (datetime.now(timezone.utc).isoformat(), patient_id),
    )
    conn.commit()
    conn.close()