# src/eval/run_full_eval.py
import sys
import io
from pathlib import Path
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

OUT_PATH = ROOT / "data" / "eval" / "evaluation_summary.md"

def run_section(title: str, func):
    print(f"\n{'='*20} {title} {'='*20}")
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            func()
    except Exception as e:
        print(f"[ERROR in {title}]: {e}")
    output = buf.getvalue()
    print(output)
    return output

def run_classifier_report():
    from nlp.baseline import train_baseline
    train_baseline()

def run_red_flag_tests():
    sys.path.insert(0, str(ROOT / "tests"))
    import test_red_flags
    test_red_flags.run_tests()

def summarize_rag_results(results):
    summary = {
        "generated": 0,
        "prepared_fallback": 0,
        "technical_failure": 0,
        "retrieval_refusal": 0,
    }
    for result in results:
        answer = result.get("answer", "")
        if result.get("answered") is False:
            summary["retrieval_refusal"] += 1
        elif "جواب جاهز مبني على معلومات موثوقة" in answer:
            summary["prepared_fallback"] += 1
        elif "مشكل تقني" in answer:
            summary["technical_failure"] += 1
        else:
            summary["generated"] += 1
    return summary

def run_rag_reliability():
    from rag.eval_reliability import TEST_QUESTIONS
    from rag.qa_agent import answer_question
    results = [answer_question(q) for q in TEST_QUESTIONS]
    summary = summarize_rag_results(results)
    total = len(results)
    print(f"RAG outcomes ({total} questions):")
    for outcome, count in summary.items():
        percentage = count / total * 100 if total else 0
        print(f"  {outcome}: {count}/{total} ({percentage:.0f}%)")

def main():
    sections = {}
    sections["Classifier (TF-IDF baseline)"] = run_section(
        "Classifier (TF-IDF baseline)", run_classifier_report
    )
    sections["Red-Flag Safety Tests"] = run_section(
        "Red-Flag Safety Tests", run_red_flag_tests
    )
    sections["RAG Reliability"] = run_section(
        "RAG Reliability", run_rag_reliability
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")
        for title, content in sections.items():
            f.write(f"## {title}\n\n```\n{content}\n```\n\n")

    print(f"\nFull summary written to {OUT_PATH}")

if __name__ == "__main__":
    main()