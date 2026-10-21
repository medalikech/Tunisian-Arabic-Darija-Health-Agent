# src/scheduler/checkin_parser.py
import re
from nlp.normalize import normalize

TAKEN_TOKENS = {
    "نعم", "ايه", "إيه", "خذيت", "خديت", "خذيتو", "خديتو",
    "ne3am", "5dit", "khdit", "oui", "yes", "took", "taken",
}
MISSED_TOKENS = {
    "لا", "لأ", "نسيت", "nsit", "non", "no", "nope", "missed",
}
NEGATED_TAKEN_PATTERNS = (
    r"\b(?:ما|مش|موش)\s*(?:خذيت|خديت|اخذت|اخذيت)\w*\b",
    r"\bma\s*(?:5dit|khdit|ne5it)\w*\b",
    r"\b(?:did\s+not|didn't|didnt|not)\s+(?:take|taken|took)\b",
)

def parse_checkin_reply(text: str) -> str | None:
    """Returns 'taken', 'missed', or None if unclear."""
    normalized = normalize(text)
    tokens = set(re.findall(r"[\w]+", normalized, flags=re.UNICODE))
    taken = bool(tokens & TAKEN_TOKENS)
    missed = bool(tokens & MISSED_TOKENS) or any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in NEGATED_TAKEN_PATTERNS
    )

    if taken and missed:
        return None
    if missed:
        return "missed"
    if taken:
        return "taken"
    return None