# src/agents/debug_syspath.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

print("sys.path entries:")
for p in sys.path:
    print(" ", p)

print("\nDoes src/storage exist?", (ROOT / "src" / "storage").exists())
print("Contents of src/storage:")
for f in (ROOT / "src" / "storage").iterdir():
    print(" ", f.name)

try:
    import storage
    print("\n'storage' package found at:", storage.__file__)
except ImportError as e:
    print("\nFailed to import storage:", e)

try:
    from storage.conversation_log import init_db
    print("conversation_log imported successfully")
except ImportError as e:
    print("Failed to import conversation_log:", e)