import base64
import binascii
import os
import sqlite3

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


KEY_ENV_VAR = "DARJA_DATA_ENCRYPTION_KEY"
ENCRYPTED_PREFIX = "enc:v1:"
NONCE_LENGTH = 12
KEY_CHECK_TEXT = "darija-health-agent-data-key-check-v1"
ENCRYPTED_COLUMNS = {
    "messages": ("raw_text", "response"),
    "escalations": ("raw_text", "reviewer_note"),
}


class EncryptionConfigurationError(RuntimeError):
    pass


def get_encryption_key() -> bytes:
    encoded_key = os.environ.get(KEY_ENV_VAR)
    if not encoded_key:
        raise EncryptionConfigurationError(
            f"{KEY_ENV_VAR} is required to read or write encrypted patient data."
        )

    try:
        key = base64.b64decode(
            encoded_key.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise EncryptionConfigurationError(
            f"{KEY_ENV_VAR} must be a URL-safe base64-encoded 32-byte key."
        ) from error

    if len(key) != 32:
        raise EncryptionConfigurationError(
            f"{KEY_ENV_VAR} must be a URL-safe base64-encoded 32-byte key."
        )
    return key


def encrypt_text(value: str | None) -> str | None:
    if value is None:
        return None

    key = get_encryption_key()
    nonce = os.urandom(NONCE_LENGTH)
    ciphertext = AESGCM(key).encrypt(nonce, value.encode("utf-8"), None)
    payload = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
    return f"{ENCRYPTED_PREFIX}{payload}"


def decrypt_text(value: str | None) -> str | None:
    if value is None or not value.startswith(ENCRYPTED_PREFIX):
        return value

    key = get_encryption_key()
    try:
        payload = base64.urlsafe_b64decode(
            value[len(ENCRYPTED_PREFIX):].encode("ascii")
        )
        if len(payload) <= NONCE_LENGTH:
            raise ValueError("Encrypted value is incomplete.")
        plaintext = AESGCM(key).decrypt(
            payload[:NONCE_LENGTH],
            payload[NONCE_LENGTH:],
            None,
        )
        return plaintext.decode("utf-8")
    except (
        InvalidTag,
        UnicodeDecodeError,
        UnicodeEncodeError,
        binascii.Error,
        ValueError,
    ) as error:
        raise ValueError("Encrypted patient data could not be authenticated.") from error


def ensure_database_key(connection: sqlite3.Connection) -> None:
    get_encryption_key()
    connection.execute(
        "CREATE TABLE IF NOT EXISTS data_encryption_metadata "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), key_check TEXT NOT NULL)"
    )
    row = connection.execute(
        "SELECT key_check FROM data_encryption_metadata WHERE id = 1"
    ).fetchone()
    if row is not None:
        try:
            check_text = decrypt_text(row[0])
        except ValueError as error:
            raise EncryptionConfigurationError(
                "The configured data-encryption key does not match this database."
            ) from error
        if check_text != KEY_CHECK_TEXT:
            raise EncryptionConfigurationError(
                "The configured data-encryption key does not match this database."
            )
        return

    for table, columns in ENCRYPTED_COLUMNS.items():
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if exists is None:
            continue
        table_columns = {
            column[1]
            for column in connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }
        for column in columns:
            if column not in table_columns:
                continue
            encrypted_row = connection.execute(
                f"SELECT {column} FROM {table} "
                f"WHERE {column} LIKE ? LIMIT 1",
                (f"{ENCRYPTED_PREFIX}%",),
            ).fetchone()
            if encrypted_row is not None:
                try:
                    decrypt_text(encrypted_row[0])
                except ValueError as error:
                    raise EncryptionConfigurationError(
                        "The configured data-encryption key does not match this database."
                    ) from error
                break

    connection.execute(
        "INSERT INTO data_encryption_metadata (id, key_check) VALUES (1, ?)",
        (encrypt_text(KEY_CHECK_TEXT),),
    )
