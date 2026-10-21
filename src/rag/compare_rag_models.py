# src/rag/compare_rag_models.py
import sys, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

MODELS_TO_TEST = ["qwen2.5:14b-instruct", "mistral:7b"]

TEST_QUESTIONS = [
    "\u0648\u0642\u062a\u0627\u0634 \u0644\u0627\u0632\u0645 \u0646\u0627\u062e\u0630 \u0627\u0644\u062f\u0648\u0627\u0621\u061f",
    "\u0634\u0646\u0648\u0629 \u0627\u0644\u0623\u0639\u0631\u0627\u0636 \u0627\u0644\u062c\u0627\u0646\u0628\u064a\u0629 \u0645\u062a\u0627\u0639 \u0647\u0630\u0627 \u0627\u0644\u062f\u0648\u0627\u0621\u061f",
    "\u0634\u0646\u0648\u0629 \u0623\u062d\u0633\u0646 \u0645\u0627\u0643\u0644\u0629 \u0641\u064a \u0627\u0644\u0639\u064a\u062f \u0627\u0644\u0643\u0628\u064a\u0631",
    "\u0643\u064a\u0641\u0627\u0634 \u0646\u062c\u062f\u062f \u0627\u0644\u0648\u0635\u0641\u0629 \u0645\u062a\u0627\u0639 \u0627\u0644\u062f\u0648\u0627\u0621",
    "\u0634\u0646\u0648\u0629 \u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0637\u0628\u064a\u0639\u064a \u0645\u062a\u0627\u0639 \u0627\u0644\u0633\u0643\u0631",
]

for model in MODELS_TO_TEST:
    print(f"\n{'='*20} Testing {model} {'='*20}")
    import os
    os.environ["RAG_LLM_MODEL"] = model

    # Re-import fresh so the module picks up the new env var
    import importlib
    import rag.qa_agent as qa_mod
    importlib.reload(qa_mod)

    real_success = 0
    for q in TEST_QUESTIONS:
        start = time.time()
        result = qa_mod.answer_question(q)
        elapsed = time.time() - start
        used_fallback = "\u062c\u0648\u0627\u0628 \u062c\u0627\u0647\u0632" in result["answer"]
        if not used_fallback:
            real_success += 1
        status = "FALLBACK" if used_fallback else "GENERATED"
        print(f"  [{status}] {elapsed:.1f}s | {q[:30]}...")

    print(f"\n{model}: {real_success}/{len(TEST_QUESTIONS)} real generation success "
          f"({real_success/len(TEST_QUESTIONS)*100:.0f}%)")