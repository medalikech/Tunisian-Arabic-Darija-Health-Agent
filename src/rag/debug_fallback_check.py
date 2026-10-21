# src/rag/debug_fallback_check.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import generate_answer

# Force a failure scenario is hard to simulate directly, so instead
# let's just check the literal failure string used internally
import rag.qa_agent as qa

# Print the exact bytes/codepoints of whatever string generate_answer
# returns on total failure, by checking the source directly
import inspect
src = inspect.getsource(qa.generate_answer)
print(src)