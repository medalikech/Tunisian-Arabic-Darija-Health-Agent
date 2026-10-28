# src/interface/cli.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from agents.router import handle_message
from patient_storage.data_crypto import get_encryption_key


def print_rag_debug(result):
    if result.get("rag_sources"):
        print(f"[debug: RAG sources={', '.join(result['rag_sources'])}]")
    if result.get("rag_trace"):
        print("[debug: RAG retrieval trace]")
        for item in result["rag_trace"]:
            print(
                "  "
                f"source={item.get('source') or '(unknown)'}, "
                f"semantic={item['semantic_score']:.3f}, "
                f"lexical_matches={item['lexical_matches']}, "
                f"coverage={item['lexical_coverage']:.3f}, "
                f"passes_gate={item['passes_relevance_gate']}, "
                f"sent_to_generation={item['sent_to_generation']}"
            )
        print(
            f"[debug: RAG answered={result.get('rag_answered', False)}, "
            f"retrieval_accepted={result.get('rag_retrieval_accepted', False)}, "
            f"answer_status={result.get('rag_answer_status', 'unknown')}]"
        )


def main():
    get_encryption_key()
    print("=" * 50)
    print("مساعد السكري - اكتب 'bye' أو 'خروج' باش توقف")
    print("=" * 50)

    patient_id = input("\nشكون انتي؟ (اسم أو رقم تعريف): ").strip() or "anonymous_patient"

    print(f"\nأهلا {patient_id}! كيفاش نجم نعاونك اليوم؟\n")

    while True:
        user_input = input("أنت: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("bye", "quit", "exit", "خروج", "وداعا"):
            print("\nمساعد السكري: صحة وعافية، تواصل معايا وقتاش تحب. بالسلامة!")
            break

        result = handle_message(user_input, patient_id=patient_id)
        print(f"\nمساعد السكري: {result['response']}\n")

        # Optional: show routing info for your own testing/debugging
        if "--debug" in sys.argv:
            print(f"[debug: route={result['route']}, intent={result.get('intent')}, "
                  f"confidence={result.get('confidence')}]\n")
            print_rag_debug(result)

if __name__ == "__main__":
    main()