# tests/test_red_flags.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from safety.red_flags import scan_for_red_flags

# (text, expected_category_or_None)
TEST_CASES = [
    ("نحس بوجع قوي في صدري من شوية", "chest_pain"),
    ("j'ai une douleur intense dans la poitrine", "chest_pain"),
    ("ما نقدرش نتنفس مليح من شوية", "breathing"),
    ("نحس برعشة وعرق بارد وخوف مفاجئ", "hypoglycemia"),
    ("ما عادش نحب نعيش هكة", "suicidal"),
    ("ما عادش نحب نعيش هكة", "suicidal"),   # space variant of ماعادش — regression test
    ("je veux en finir", "suicidal"),
    ("خذيت جرعة زايدة بالغلط من الدواء", "overdose"),
    ("nsit nechreb ed-dwa lyoum", None),          # should NOT trigger
    ("شنوة الأكلات لي لازم نبعد عليهم", None),      # should NOT trigger
    ("سلام كيفاش الحال", None),                     # should NOT trigger
    # --- Original basic cases ---
    ("نحس بوجع قوي في صدري من شوية", "chest_pain"),
    ("j'ai une douleur intense dans la poitrine", "chest_pain"),
    ("ما نقدرش نتنفس مليح من شوية", "breathing"),
    ("نحس برعشة وعرق بارد وخوف مفاجئ", "hypoglycemia"),
    ("ما عادش نحب نعيش هكة", "suicidal"),
    ("je veux en finir", "suicidal"),
    ("خذيت جرعة زايدة بالغلط من الدواء", "overdose"),
    ("nsit nechreb ed-dwa lyoum", None),
    ("شنوة الأكلات لي لازم نبعد عليهم", None),
    ("سلام كيفاش الحال", None),

    # --- Adversarial: should trigger but phrased differently/harder ---
    ("صدري يوجعني برشا و مانيش قادر نتنفس مليح", "chest_pain"),  # mentions both chest+breathing, either acceptable
    ("نحس بضغط قوي في صدري توا بالضبط", "chest_pain"),
    ("خناق في صدري من دقيقتين و الوجع يزيد", "breathing"),
    ("رجليا ترعشو و عرقان برد بزاف", "hypoglycemia"),
    ("ماعندیش حتى سبب نكمل نعيش", "suicidal"),  # alt phrasing of hopelessness
    ("حابة نرتاح من كل شيء نهائيا", "suicidal"),   # euphemistic, harder
    ("غلطت و خذيت الدواء زوز مرات اليوم", "overdose"),

    # --- Adversarial: should NOT trigger (past tense / resolved / unrelated) ---
    ("العام الفارط كان عندي وجع في صدري بس توة ولا باهي", None),  # PAST, resolved — tricky!
    ("صاحبي قالي عندو مشكل في التنفس", None),  # about someone ELSE, not the patient
    ("شفت فيلم اليوم على واحد عندو نوبة قلبية", None),  # unrelated, about media
    ("نحب نعيش حياة صحية أكثر و نتحسن", None),  # contains "نعيش" but positive context
    ("خذيت الدواء مرة وحدة بس اليوم كيف العادة", None),  # NOT a double dose
    ("كيفاش نعرف إذا عندي خطر الإصابة بأزمة قلبية في المستقبل", None),  # informational question, not current symptom
]

def run_tests():
    passed, failed = 0, 0
    for text, expected in TEST_CASES:
        matches = scan_for_red_flags(text)
        got = matches[0] if matches else None
        ok = (got == expected) if expected else (got is None)
        status = "PASS" if ok else "FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"[{status}] expected={expected} got={got} | {text}")
    print(f"\n{passed}/{passed+failed} passed")

if __name__ == "__main__":
    run_tests()