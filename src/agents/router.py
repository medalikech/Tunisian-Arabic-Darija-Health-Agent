# src/agents/router.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from safety.red_flags import scan_for_red_flags
from safety.emergency_responses import get_emergency_response
from patient_storage.conversation_log import init_db, log_interaction
import joblib
from nlp.normalize import normalize
init_db()
CLASSIFIER = joblib.load(ROOT / "models" / "baseline_tfidf.joblib")

# Intents that should be answered by the RAG Q&A agent
RAG_INTENTS = {"ask_medication_info", "ask_side_effect", "ask_diet_lifestyle", "ask_refill"}
LOW_CONFIDENCE_THRESHOLD = 0.15
# Intents with simple fixed/templated responses (no LLM needed)
FIXED_RESPONSES = {
"greeting_smalltalk": "أهلا وسهلا! كيفاش نجم نعاونك اليوم؟",
    "missed_dose": "ما تهلعش، خذ الجرعة التالية في وقتها العادي. ما تضاعفش الجرعة أبدا. كان عندك شك، تواصل مع طبيبك.",
    "emotional_support": "نحس بيك، وما نحبش تكون وحدك في هذا. هذا الشي صعيب فعلا وحق عليك تحس بيه. تحب تحكيلي أكثر شنوة لي قاعد يضايقك؟",
    "want_to_stop_med": "نفهم إنك تعبت من الدواء، وهذا حق طبيعي. بس وقف الدواء وحدك بلا استشارة طبيبك ممكن يكون خطير. أحسنلك تحكي مع طبيبك قبل أي قرار.",
    "report_symptom": "شكراً إنك حكيتلي على هذا. كان الأعراض قوية أو مستمرة، لازم تتواصل مع طبيبك. تحب تزيد تفاصيل أكثر؟",
    "report_glucose_value": "شكراً على المعلومة. كان النتيجة خارجة عن المعدل العادي بصفة متكررة، لازم تتواصل مع طبيبك لمتابعة العلاج.",
}

def handle_message(raw_text: str, patient_id: str = "test_patient") -> dict:
    # STEP 1: Rule-based safety check — always first
    red_flags = scan_for_red_flags(raw_text)
    if red_flags:
        category = red_flags[0]
        result = {
            "route": "emergency",
            "matched_categories": red_flags,
            "response": get_emergency_response(category),
            "classifier_bypassed": True,
        }
        log_interaction(patient_id, raw_text, result)
        return result

    # STEP 2: Intent classification
    normalized = normalize(raw_text)
    intent = CLASSIFIER.predict([normalized])[0]
    confidence = max(CLASSIFIER.predict_proba([normalized])[0])
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        result = {
            "route": "clarify",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": "معذرة، ما فهمتش بالضبط شنوة تقصد. تنجم تعطيني تفاصيل أكثر؟ مثلا: نسيت الدواء، عندك وجع، ولا تحب تسولني على حاجة؟",
        }
    log_interaction(patient_id, raw_text, result)
    return result
    # STEP 3: Route based on intent
    if intent in FIXED_RESPONSES:
        result = {
            "route": "fixed_response",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": FIXED_RESPONSES[intent],
        }
        log_interaction(patient_id, raw_text, result)
        return result

    if intent in RAG_INTENTS:
        from rag.qa_agent import answer_question
        rag_result = answer_question(raw_text)
        result = {
            "route": "rag",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": rag_result["answer"],
            "rag_score": rag_result.get("best_score"),
        }
        log_interaction(patient_id, raw_text, result)
        return result

    result = {
        "route": "unhandled",
        "intent": intent,
        "confidence": round(float(confidence), 3),
        "response": "فهمتك، ونحب نعاونك أكثر في هذا الموضوع. (هذا الجزء مازال قيد التطوير)",
    }
    log_interaction(patient_id, raw_text, result)
    return result

if __name__ == "__main__":
    test_messages = [
        "نحس بوجع قوي في صدري من شوية",       # should hit emergency
        "nsit nechreb ed-dwa lyoum",           # should hit fixed_response
        "وقتاش لازم ناخذ الدواء",              # should hit rag
        "سلام كيفاش الحال",                    # should hit fixed_response
        "نحس بالضيق من هذا المرض",              # should hit unhandled (for now)
    ]
    for msg in test_messages:
        result = handle_message(msg)
        print(f"\nInput: {msg}")
        print(f"Route: {result['route']}")
        print(f"Response: {result['response']}")