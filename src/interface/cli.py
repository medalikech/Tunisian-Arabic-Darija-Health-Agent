# src/interface/cli.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from agents.router import handle_message

def main():
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

if __name__ == "__main__":
    main()