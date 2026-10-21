import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parents[2]
AUTH_DB_PATH = ROOT / "data" / "auth.db"
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024
MAX_USERNAME_LENGTH = 128
MAX_PATIENT_ID_LENGTH = 128
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60


def _connect():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_auth_db():
    with _connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('patient', 'staff')),
                patient_id TEXT UNIQUE,
                CHECK (
                    (role = 'patient' AND patient_id IS NOT NULL) OR
                    (role = 'staff' AND patient_id IS NULL)
                )
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS staff_patient_access (
                staff_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                patient_id TEXT NOT NULL,
                PRIMARY KEY (staff_user_id, patient_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_failures (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                failed_attempts INTEGER NOT NULL,
                locked_until INTEGER
            )
        """)


def create_user(username, password, role, patient_id=None):
    username = username.strip() if isinstance(username, str) else ""
    if not username:
        raise ValueError("Username must not be empty.")
    if len(username) > MAX_USERNAME_LENGTH:
        raise ValueError(f"Username must be at most {MAX_USERNAME_LENGTH} characters.")
    if (
        not isinstance(password, str)
        or len(password) < MIN_PASSWORD_LENGTH
        or len(password) > MAX_PASSWORD_LENGTH
    ):
        raise ValueError(
            f"Password must be between {MIN_PASSWORD_LENGTH} and "
            f"{MAX_PASSWORD_LENGTH} characters long."
        )
    if role not in ("patient", "staff"):
        raise ValueError("Role must be 'patient' or 'staff'.")
    if role == "patient" and (not isinstance(patient_id, str) or not patient_id.strip()):
        raise ValueError("A patient account must have a patient ID.")
    if role == "staff" and patient_id is not None:
        raise ValueError("A staff account cannot have a patient ID.")

    patient_id = patient_id.strip() if patient_id else None
    if patient_id and len(patient_id) > MAX_PATIENT_ID_LENGTH:
        raise ValueError(f"Patient ID must be at most {MAX_PATIENT_ID_LENGTH} characters.")
    password_hash = generate_password_hash(password, method="scrypt")
    try:
        with _connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, patient_id) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, role, patient_id),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username or patient ID is already registered.") from exc
    return {"id": user_id, "username": username, "role": role, "patient_id": patient_id}


def authenticate_user(username, password):
    if not isinstance(username, str) or not isinstance(password, str):
        return None
    with _connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, username, password_hash, role, patient_id "
            "FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if row is None:
            return None
        failure = conn.execute(
            "SELECT failed_attempts, locked_until FROM login_failures "
            "WHERE user_id = ?",
            (row["id"],),
        ).fetchone()
    now = int(time.time())
    if failure is not None and failure["locked_until"] is not None:
        if failure["locked_until"] > now:
            return None
    if not check_password_hash(row["password_hash"], password):
        _record_login_failure(row["id"], now)
        return None
    with _connection() as conn:
        conn.execute("DELETE FROM login_failures WHERE user_id = ?", (row["id"],))
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "patient_id": row["patient_id"],
    }


def _record_login_failure(user_id, now):
    with _connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT failed_attempts, locked_until FROM login_failures "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is not None and row[1] is not None and row[1] > now:
            return

        failed_attempts = row[0] if row is not None and row[1] is None else 0
        failed_attempts += 1
        locked_until = (
            now + LOGIN_LOCKOUT_SECONDS
            if failed_attempts >= MAX_LOGIN_ATTEMPTS
            else None
        )
        conn.execute(
            "INSERT INTO login_failures (user_id, failed_attempts, locked_until) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "failed_attempts = excluded.failed_attempts, "
            "locked_until = excluded.locked_until",
            (user_id, failed_attempts, locked_until),
        )


def get_user(user_id):
    with _connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, username, role, patient_id FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def assign_patient_to_staff(staff_username, patient_id):
    with _connection() as conn:
        staff = conn.execute(
            "SELECT id FROM users WHERE username = ? AND role = 'staff'",
            (staff_username.strip(),),
        ).fetchone()
        patient = conn.execute(
            "SELECT 1 FROM users WHERE patient_id = ? AND role = 'patient'",
            (patient_id.strip(),),
        ).fetchone()
        if staff is None:
            raise ValueError("Staff account was not found.")
        if patient is None:
            raise ValueError("Patient account was not found.")
        conn.execute(
            "INSERT OR IGNORE INTO staff_patient_access (staff_user_id, patient_id) "
            "VALUES (?, ?)",
            (staff[0], patient_id.strip()),
        )


def can_access_patient(user, patient_id):
    if not user or not isinstance(patient_id, str):
        return False
    if user["role"] == "patient":
        return user["patient_id"] == patient_id
    if user["role"] != "staff":
        return False
    with _connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM staff_patient_access "
            "WHERE staff_user_id = ? AND patient_id = ?",
            (user["id"], patient_id),
        ).fetchone()
    return row is not None


def get_accessible_patient_ids(staff_user_id):
    with _connection() as conn:
        rows = conn.execute(
            "SELECT patient_id FROM staff_patient_access "
            "WHERE staff_user_id = ? ORDER BY patient_id",
            (staff_user_id,),
        ).fetchall()
    return [row[0] for row in rows]
