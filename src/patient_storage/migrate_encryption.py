import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from patient_storage.data_crypto import (
    ENCRYPTED_PREFIX,
    decrypt_text,
    encrypt_text,
    ensure_database_key,
    get_encryption_key,
)
from patient_storage.conversation_log import DB_PATH


TEXT_COLUMNS = {
    "messages": ("raw_text", "response"),
    "escalations": ("raw_text", "reviewer_note"),
}


def _existing_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if exists is None:
        return set()
    return {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _create_backup(database_path: Path, backup_path: Path) -> None:
    database_path = database_path.resolve()
    backup_path = backup_path.resolve()
    if database_path == backup_path:
        raise ValueError("Backup path must be different from the database path.")
    if backup_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with closing(sqlite3.connect(database_path)) as source:
            with closing(sqlite3.connect(backup_path)) as backup:
                source.backup(backup)
                backup.commit()
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise


def _encrypt_existing_value(value: str | None) -> str | None:
    if isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX):
        decrypt_text(value)
        return value
    return encrypt_text(value)


def migrate_database(database_path: Path, backup_path: Path) -> int:
    get_encryption_key()
    database_path = Path(database_path)
    backup_path = Path(backup_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")

    _create_backup(database_path, backup_path)
    updated_count = 0

    with closing(sqlite3.connect(database_path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            for table, candidate_columns in TEXT_COLUMNS.items():
                columns = _existing_columns(connection, table)
                encrypted_columns = [
                    column for column in candidate_columns if column in columns
                ]
                if not encrypted_columns or "id" not in columns:
                    continue

                rows = connection.execute(
                    f"SELECT id, {', '.join(encrypted_columns)} FROM {table}"
                ).fetchall()
                for row in rows:
                    values = []
                    for value in row[1:]:
                        encrypted = _encrypt_existing_value(value)
                        values.append(encrypted)
                        if (
                            isinstance(value, str)
                            and not value.startswith(ENCRYPTED_PREFIX)
                        ):
                            updated_count += 1
                    assignments = ", ".join(
                        f"{column} = ?" for column in encrypted_columns
                    )
                    connection.execute(
                        f"UPDATE {table} SET {assignments} WHERE id = ?",
                        (*values, row[0]),
                    )
            ensure_database_key(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt existing patient text fields in the conversation database."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DB_PATH,
        help=f"SQLite database to migrate (default: {DB_PATH})",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help="New backup path; the migration refuses to overwrite an existing file.",
    )
    args = parser.parse_args()

    count = migrate_database(args.database, args.backup)
    print(f"Encrypted {count} plaintext values.")
    print(f"Backup created at {args.backup.resolve()}")
    print("Keep the backup and encryption key in secure, separate locations.")


if __name__ == "__main__":
    main()
