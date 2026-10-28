# src/webapp/view_my_chat.py
from pathlib import Path
import sys
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from patient_storage.conversation_log import DB_PATH
from patient_storage.data_crypto import decrypt_text

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT * FROM messages WHERE patient_id LIKE 'web_%' ORDER BY timestamp ASC"
).fetchall()
conn.close()

for r in rows:
    print(f"\n[{r['patient_id']}] {r['timestamp']}")
    print(f"  You: {decrypt_text(r['raw_text'])}")
    print(f"  Bot ({r['route']}): {decrypt_text(r['response'])}")