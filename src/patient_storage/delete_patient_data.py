import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from patient_storage.conversation_log import DB_PATH as CONVERSATIONS_DB_PATH
from webapp.auth import AUTH_DB_PATH


CONVERSATION_TABLES = (
    "messages",
    "dose_events",
    "checkin_schedule",
)


def _table_exists(connection: sqlite3.Connection, database: str, table: str) -> bool:
    row = connection.execute(
        f"SELECT 1 FROM {database}.sqlite_master "
        "WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _delete_rows(
    connection: sqlite3.Connection,
    database: str,
    table: str,
    patient_id: str,
) -> int:
    if not _table_exists(connection, database, table):
        return 0
    cursor = connection.execute(
        f"DELETE FROM {database}.{table} WHERE patient_id = ?",
        (patient_id,),
    )
    return max(cursor.rowcount, 0)


def delete_patient_data(
    patient_id: str,
    confirm_patient_id: str,
    conversations_db_path: Path = CONVERSATIONS_DB_PATH,
    auth_db_path: Path = AUTH_DB_PATH,
) -> dict[str, int]:
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("Patient ID must not be empty.")
    if confirm_patient_id != patient_id:
        raise ValueError("Confirmation must exactly match the patient ID.")

    conversations_db_path = Path(conversations_db_path)
    auth_db_path = Path(auth_db_path)
    if not conversations_db_path.is_file():
        raise FileNotFoundError(
            f"Conversation database not found: {conversations_db_path}"
        )
    if not auth_db_path.is_file():
        raise FileNotFoundError(f"Authentication database not found: {auth_db_path}")
    if conversations_db_path.resolve() == auth_db_path.resolve():
        raise ValueError("Conversation and authentication databases must be separate.")

    with closing(sqlite3.connect(conversations_db_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("ATTACH DATABASE ? AS auth_db", (str(auth_db_path),))
        try:
            journal_modes = {
                connection.execute("PRAGMA main.journal_mode").fetchone()[0].lower(),
                connection.execute("PRAGMA auth_db.journal_mode").fetchone()[0].lower(),
            }
            if "wal" in journal_modes or "memory" in journal_modes:
                raise RuntimeError(
                    "Deletion refused: both databases must use a rollback journal "
                    "for a coordinated transaction."
                )

            counts = {}
            connection.execute("BEGIN IMMEDIATE")
            for table in CONVERSATION_TABLES:
                counts[table] = _delete_rows(
                    connection,
                    "main",
                    table,
                    patient_id,
                )

            counts["staff_patient_access"] = _delete_rows(
                connection,
                "auth_db",
                "staff_patient_access",
                patient_id,
            )

            if _table_exists(connection, "main", "escalations"):
                counts["escalations_retained"] = connection.execute(
                    "SELECT COUNT(*) FROM main.escalations WHERE patient_id = ?",
                    (patient_id,),
                ).fetchone()[0]
            else:
                counts["escalations_retained"] = 0

            if _table_exists(connection, "auth_db", "users"):
                patient_row = connection.execute(
                    "SELECT id FROM auth_db.users "
                    "WHERE role = 'patient' AND patient_id = ?",
                    (patient_id,),
                ).fetchone()
                if patient_row is None:
                    counts["patient_accounts"] = 0
                else:
                    cursor = connection.execute(
                        "DELETE FROM auth_db.users WHERE id = ?",
                        (patient_row[0],),
                    )
                    counts["patient_accounts"] = max(cursor.rowcount, 0)
            else:
                counts["patient_accounts"] = 0

            connection.commit()
            return counts
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("DETACH DATABASE auth_db")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete a patient's messages, adherence/check-in data, account, "
            "and access links. Emergency escalation records are retained."
        )
    )
    parser.add_argument("--patient-id", required=True)
    parser.add_argument(
        "--confirm-patient-id",
        required=True,
        help="Repeat the exact patient ID to confirm this deletion.",
    )
    args = parser.parse_args()

    try:
        counts = delete_patient_data(
            args.patient_id,
            args.confirm_patient_id,
        )
    except (FileNotFoundError, RuntimeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))

    print(f"Deleted data for patient ID {args.patient_id!r}:")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    print("Emergency escalation records were retained.")


if __name__ == "__main__":
    main()
