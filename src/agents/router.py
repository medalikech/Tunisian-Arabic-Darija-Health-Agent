# src/agents/router.py
from pathlib import Path
import logging
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from patient_storage.adherence_log import init_adherence_db, log_dose_event
from safety.red_flags import scan_for_red_flags
from safety.emergency_responses import get_emergency_response
from patient_storage.conversation_log import init_db, log_interaction, get_recent_context
import joblib
from nlp.normalize import normalize
from scheduler.checkin_parser import parse_checkin_reply
from patient_storage.adherence_log import init_adherence_db, log_dose_event
from patient_storage.escalation_log import init_escalation_db, log_escalation

logger = logging.getLogger(__name__)

init_db()
init_adherence_db()
init_escalation_db()
CLASSIFIER = joblib.load(ROOT / "models" / "baseline_merged.joblib")

RAG_INTENTS = {"ask_medication_info", "ask_side_effect", "ask_diet_lifestyle", "ask_refill"}
CHECKIN_UNCLEAR_RESPONSE = (
    "\u0645\u0627 \u062d\u0628\u064a\u062a\u0634 \u0646\u0641\u0647\u0645 \u063a\u0627\u0644\u0637. "
    "\u062e\u0630\u064a\u062a \u0627\u0644\u062f\u0648\u0627\u061f "
    "\u062c\u0627\u0648\u0628 \u0628\u0646\u0639\u0645 \u0623\u0648 \u0644\u0627."
)
CHECKIN_MISSED_RESPONSE = (
    "\u0646\u0641\u0647\u0645\u0643. \u0645\u0627 \u0646\u0646\u062c\u0645\u0634 "
    "\u0646\u062d\u062f\u062f\u0644\u0643 \u0648\u0642\u062a\u0627\u0634 "
    "\u062a\u0627\u062e\u0630 \u0627\u0644\u062c\u0631\u0639\u0629 "
    "\u0627\u0644\u0646\u0627\u0642\u0635\u0629. \u062a\u062b\u0628\u062a "
    "\u0645\u0646 \u0627\u0644\u0646\u0634\u0631\u0629 \u0627\u0644\u0637\u0628\u064a\u0629 "
    "\u0623\u0648 \u0627\u0633\u0623\u0644 \u0627\u0644\u0635\u064a\u062f\u0644\u064a "
    "\u0623\u0648 \u0637\u0628\u064a\u0628\u0643 \u0642\u0628\u0644 "
    "\u0645\u0627 \u062a\u062a\u0635\u0631\u0641."
)
LOW_CONFIDENCE_THRESHOLD = 0.15

FIXED_RESPONSES = {
    "greeting_smalltalk": "\u0623\u0647\u0644\u0627 \u0648\u0633\u0647\u0644\u0627! \u0643\u064a\u0641\u0627\u0634 \u0646\u062c\u0645 \u0646\u0639\u0627\u0648\u0646\u0643 \u0627\u0644\u064a\u0648\u0645\u061f",
    "missed_dose": "\u0646\u0641\u0647\u0645\u0643. \u0645\u0627 \u0646\u0646\u062c\u0645\u0634 \u0646\u062d\u062f\u062f\u0644\u0643 \u0648\u0642\u062a\u0627\u0634 \u062a\u0627\u062e\u0630 \u0627\u0644\u062c\u0631\u0639\u0629 \u0627\u0644\u0646\u0627\u0642\u0635\u0629. \u062a\u062b\u0628\u062a \u0645\u0646 \u0627\u0644\u0646\u0634\u0631\u0629 \u0627\u0644\u0637\u0628\u064a\u0629 \u0623\u0648 \u0627\u0633\u0623\u0644 \u0627\u0644\u0635\u064a\u062f\u0644\u064a \u0623\u0648 \u0637\u0628\u064a\u0628\u0643 \u0642\u0628\u0644 \u0645\u0627 \u062a\u062a\u0635\u0631\u0641.",
    "emotional_support": "\u0646\u062d\u0633 \u0628\u064a\u0643\u060c \u0648\u0645\u0627 \u0646\u062d\u0628\u0634 \u062a\u0643\u0648\u0646 \u0648\u062d\u062f\u0643 \u0641\u064a \u0647\u0630\u0627. \u0647\u0630\u0627 \u0627\u0644\u0634\u064a \u0635\u0639\u064a\u0628 \u0641\u0639\u0644\u0627 \u0648\u062d\u0642 \u0639\u0644\u064a\u0643 \u062a\u062d\u0633 \u0628\u064a\u0647. \u062a\u062d\u0628 \u062a\u062d\u0643\u064a\u0644\u064a \u0623\u0643\u062b\u0631 \u0634\u0646\u0648\u0629 \u0644\u064a \u0642\u0627\u0639\u062f \u064a\u0636\u0627\u064a\u0642\u0643\u061f",
    "want_to_stop_med": "\u0646\u0641\u0647\u0645 \u0625\u0646\u0643 \u062a\u0639\u0628\u062a \u0645\u0646 \u0627\u0644\u062f\u0648\u0627\u0621\u060c \u0648\u0647\u0630\u0627 \u062d\u0642 \u0637\u0628\u064a\u0639\u064a. \u0628\u0633 \u0648\u0642\u0641 \u0627\u0644\u062f\u0648\u0627\u0621 \u0648\u062d\u062f\u0643 \u0628\u0644\u0627 \u0627\u0633\u062a\u0634\u0627\u0631\u0629 \u0637\u0628\u064a\u0628\u0643 \u0645\u0645\u0643\u0646 \u064a\u0643\u0648\u0646 \u062e\u0637\u064a\u0631. \u0623\u062d\u0633\u0646\u0644\u0643 \u062a\u062d\u0643\u064a \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0642\u0628\u0644 \u0623\u064a \u0642\u0631\u0627\u0631.",
    "report_symptom": "\u0634\u0643\u0631\u0627\u064b \u0625\u0646\u0643 \u062d\u0643\u064a\u062a\u0644\u064a \u0639\u0644\u0649 \u0647\u0630\u0627. \u0643\u0627\u0646 \u0627\u0644\u0623\u0639\u0631\u0627\u0636 \u0642\u0648\u064a\u0629 \u0623\u0648 \u0645\u0633\u062a\u0645\u0631\u0629\u060c \u0644\u0627\u0632\u0645 \u062a\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u0637\u0628\u064a\u0628\u0643. \u062a\u062d\u0628 \u062a\u0632\u064a\u062f \u062a\u0641\u0627\u0635\u064a\u0644 \u0623\u0643\u062b\u0631\u061f",
    "report_glucose_value": "\u0634\u0643\u0631\u0627\u064b \u0639\u0644\u0649 \u0627\u0644\u0645\u0639\u0644\u0648\u0645\u0629. \u0643\u0627\u0646 \u0627\u0644\u0646\u062a\u064a\u062c\u0629 \u062e\u0627\u0631\u062c\u0629 \u0639\u0646 \u0627\u0644\u0645\u0639\u062f\u0644 \u0627\u0644\u0639\u0627\u062f\u064a \u0628\u0635\u0641\u0629 \u0645\u062a\u0643\u0631\u0631\u0629\u060c \u0644\u0627\u0632\u0645 \u062a\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0644\u0645\u062a\u0627\u0628\u0639\u0629 \u0627\u0644\u0639\u0644\u0627\u062c.",
}
FOLLOWUP_RESPONSES = {
    "report_symptom": "\u0634\u0643\u0631\u0627\u064b \u0639\u0644\u0649 \u0627\u0644\u062a\u0648\u0636\u064a\u062d. \u0628\u0645\u0627 \u0625\u0646 \u0627\u0644\u0623\u0639\u0631\u0627\u0636 \u0645\u0633\u062a\u0645\u0631\u0629 \u0623\u0648 \u0642\u0648\u064a\u0629\u060c \u0646\u0635\u062d\u0643 \u0628\u0634\u062f\u0629 \u062a\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0627\u0644\u064a\u0648\u0645 \u0646\u0641\u0633\u0647 \u0648\u0644\u0627 \u062a\u0633\u062a\u0646\u0649.",
    "emotional_support": "\u0634\u0643\u0631\u0627\u064b \u0625\u0646\u0643 \u0634\u0627\u0631\u0643\u062a\u0646\u064a \u0647\u0630\u0627. \u0647\u0630\u0627 \u0627\u0644\u0634\u064a \u0645\u0647\u0645 \u0648\u0645\u0627 \u0644\u0627\u0632\u0645\u0634 \u062a\u0648\u0627\u062c\u0647\u0647 \u0648\u062d\u062f\u0643\u060c \u062a\u062d\u0628 \u0646\u0639\u0637\u064a\u0643 \u0631\u0642\u0645 \u062e\u0637 \u0627\u0644\u0625\u0635\u063a\u0627\u0621 \u0627\u0644\u0646\u0641\u0633\u064a \u0623\u0648 \u062a\u0641\u0636\u0644 \u062a\u062a\u0643\u0644\u0645 \u0645\u0639 \u0637\u0628\u064a\u0628\u0643\u061f",
    "want_to_stop_med": "\u0646\u0641\u0647\u0645\u0643. \u0642\u0628\u0644 \u0645\u0627 \u062a\u0642\u0631\u0631\u060c \u0646\u0635\u062d\u0643 \u062a\u062d\u062f\u062f \u0645\u0648\u0639\u062f \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0628\u0627\u0644\u0642\u0631\u064a\u0628 \u062a\u062d\u0643\u064a\u0644\u0648 \u0628\u0627\u0644\u0636\u0628\u0637 \u0634\u0646\u0648\u0629 \u062d\u0627\u0633\u0633.",
    "report_glucose_value": "\u0645\u0641\u0647\u0648\u0645. \u0643\u0627\u0646 \u0647\u0630\u0627 \u064a\u062a\u0643\u0631\u0631\u060c \u0623\u062d\u0633\u0646\u0644\u0643 \u062a\u062d\u0643\u064a \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0628\u0627\u0644\u062a\u0641\u0627\u0635\u064a\u0644 \u0647\u0630\u0647 \u0628\u0627\u0644\u0636\u0628\u0637.",
}
SOFT_ESCALATION_FOLLOWUP = "\u0645\u0641\u0647\u0648\u0645. \u0628\u0645\u0627 \u0625\u0646 \u0627\u0644\u0639\u0631\u0636 \u0645\u0633\u062a\u0645\u0631\u060c \u0645\u0646 \u0627\u0644\u0623\u0641\u0636\u0644 \u062a\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0627\u0644\u064a\u0648\u0645 \u0646\u0641\u0633\u0647 \u0628\u0627\u0634 \u064a\u0634\u0648\u0641\u0643 \u0639\u0646 \u0642\u0631\u0628."

def _is_repeated_soft_escalation(patient_id):
    context = get_recent_context(patient_id, n_turns=1)
    if not context:
        return False
    return context[-1]["route"] == "soft_escalation"
def _is_repeated_followup(patient_id, intent):
    """Check if our last message to this patient was EITHER the original
    fixed-response template OR the follow-up template for this intent,
    meaning the patient is continuing an existing thread, not starting a
    new one."""
    context = get_recent_context(patient_id, n_turns=1)
    if not context:
        return False
    last_turn = context[-1]
    if last_turn["route"] not in ("fixed_response", "fixed_response_followup"):
        return False
    possible_prior_responses = {
        FIXED_RESPONSES.get(intent),
        FOLLOWUP_RESPONSES.get(intent),
    }
    return last_turn["response"] in possible_prior_responses
def _last_turn_was_rag(patient_id):
    context = get_recent_context(patient_id, n_turns=1)
    if not context:
        return False
    return context[-1]["route"] == "rag"
def handle_message(raw_text, patient_id="test_patient"):
    red_flags = scan_for_red_flags(raw_text)
    if red_flags:
        category = red_flags[0]
        result = {
            "route": "emergency",
            "matched_categories": red_flags,
            "response": get_emergency_response(category),
            "classifier_bypassed": True,
        }
        try:
            log_escalation(
                patient_id,
                raw_text,
                severity="emergency",
                category=category,
            )
        except sqlite3.Error as error:
            result["escalation_logged"] = False
            result["response"] += (
                "\n\nتنبيه: التطبيق ما سجلش الحالة وما بعثش إشعار لطبيب "
                "أو لفريق رعاية. اتصل بخدمات الطوارئ المحلية توا."
            )
            logger.error(
                "Could not record emergency escalation "
                "(category=%s, error_type=%s)",
                category,
                type(error).__name__,
            )
        else:
            result["escalation_logged"] = True

        try:
            log_interaction(patient_id, raw_text, result)
        except sqlite3.Error as error:
            result["interaction_logged"] = False
            logger.error(
                "Could not record emergency interaction "
                "(category=%s, error_type=%s)",
                category,
                type(error).__name__,
            )
        else:
            result["interaction_logged"] = True
        return result

    context = get_recent_context(patient_id, n_turns=1)
    if context and context[-1]["route"] == "proactive_checkin":
        reply = parse_checkin_reply(raw_text)
        if reply:
            log_dose_event(patient_id, reply, source="checkin_reply")
            response_text = (
                "\u0628\u0627\u0647\u064a\u060c \u0635\u062d\u0629 \u0648\u0639\u0627\u0641\u064a\u0629! \u0646\u0634\u0648\u0641\u0648\u0643 \u063a\u062f\u0648\u0629."
                if reply == "taken" else
                CHECKIN_MISSED_RESPONSE
            )
            result = {
                "route": "checkin_reply",
                "intent": None,
                "confidence": None,
                "response": response_text,
            }
            log_interaction(patient_id, raw_text, result)
            return result
        result = {
            "route": "checkin_unclear",
            "intent": None,
            "confidence": None,
            "response": CHECKIN_UNCLEAR_RESPONSE,
        }
        log_interaction(patient_id, raw_text, result)
        return result

    context = get_recent_context(patient_id, n_turns=3)
    context_text = " ".join(c["raw_text"] for c in context)
    classification_input = f"{context_text} {raw_text}".strip() if context else raw_text

    normalized = normalize(classification_input)
    intent = CLASSIFIER.predict([normalized])[0]
    confidence = max(CLASSIFIER.predict_proba([normalized])[0])
    if confidence < LOW_CONFIDENCE_THRESHOLD and _last_turn_was_rag(patient_id) and len(raw_text.split()) <= 4:
        from rag.qa_agent import answer_question
        contextual_question = f"{context_text} {raw_text}".strip()
        rag_result = answer_question(contextual_question)
        result = {
            "route": "rag_followup",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": rag_result["answer"],
            "rag_score": rag_result.get("best_score"),
            "rag_sources": rag_result.get("source_files", []),
        }
        log_interaction(patient_id, raw_text, result)
        return result
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        result = {
            "route": "clarify",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": "\u0645\u0639\u0630\u0631\u0629\u060c \u0645\u0627 \u0641\u0647\u0645\u062a\u0634 \u0628\u0627\u0644\u0636\u0628\u0637 \u0634\u0646\u0648\u0629 \u062a\u0642\u0635\u062f. \u062a\u0646\u062c\u0645 \u062a\u0639\u0637\u064a\u0646\u064a \u062a\u0641\u0627\u0635\u064a\u0644 \u0623\u0643\u062b\u0631\u061f \u0645\u062b\u0644\u0627: \u0646\u0633\u064a\u062a \u0627\u0644\u062f\u0648\u0627\u0621\u060c \u0639\u0646\u062f\u0643 \u0648\u062c\u0639\u060c \u0648\u0644\u0627 \u062a\u062d\u0628 \u062a\u0633\u0648\u0644\u0646\u064a \u0639\u0644\u0649 \u062d\u0627\u062c\u0629\u061f",
        }
        log_interaction(patient_id, raw_text, result)
        return result

    if intent in FIXED_RESPONSES:
        if intent == "missed_dose":
            log_dose_event(patient_id, "missed", source="conversation")

        if _is_repeated_followup(patient_id, intent) and intent in FOLLOWUP_RESPONSES:
            response_text = FOLLOWUP_RESPONSES[intent]
            route_label = "fixed_response_followup"
        else:
            response_text = FIXED_RESPONSES[intent]
            route_label = "fixed_response"

        result = {
            "route": route_label,
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": response_text,
        }
        log_interaction(patient_id, raw_text, result)
        return result

    if intent in RAG_INTENTS:
        from rag.qa_agent import answer_question
        contextual_question = f"{context_text} {raw_text}".strip() if context else raw_text
        rag_result = answer_question(contextual_question)
        result = {
            "route": "rag",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": rag_result["answer"],
            "rag_score": rag_result.get("best_score"),
            "rag_sources": rag_result.get("source_files", []),
        }
        log_interaction(patient_id, raw_text, result)
        return result
    if intent == "red_flag_emergency":
        log_escalation(patient_id, raw_text, severity="soft", category="classifier_flagged")
        if _is_repeated_soft_escalation(patient_id):
                response_text = SOFT_ESCALATION_FOLLOWUP
        else:
                response_text = "\u0646\u062d\u0633 \u0625\u0646 \u0647\u0630\u0627 \u0642\u062f \u064a\u0643\u0648\u0646 \u0645\u0647\u0645\u060c \u0648\u0627\u0634 \u0639\u0646\u062f\u0643 \u0623\u0639\u0631\u0627\u0636 \u062a\u062d\u0633 \u0628\u064a\u0647\u0627 \u0642\u0648\u064a\u0629 \u0623\u0648 \u0645\u0642\u0644\u0642\u0629\u061f \u0643\u0627\u0646 \u0646\u0639\u0645\u060c \u0623\u062d\u0633\u0646\u0644\u0643 \u062a\u062a\u0648\u0627\u0635\u0644 \u0645\u0639 \u0637\u0628\u064a\u0628\u0643 \u0623\u0648 \u0627\u0644\u0625\u0633\u0639\u0627\u0641 \u0641\u0648\u0631\u0627."
        result = {
            "route": "soft_escalation",
            "intent": intent,
            "confidence": round(float(confidence), 3),
            "response": response_text,
        }
        log_interaction(patient_id, raw_text, result)
        return result
    result = {
        "route": "unhandled",
        "intent": intent,
        "confidence": round(float(confidence), 3),
        "response": "\u0641\u0647\u0645\u062a\u0643\u060c \u0648\u0646\u062d\u0628 \u0646\u0639\u0627\u0648\u0646\u0643 \u0623\u0643\u062b\u0631 \u0641\u064a \u0647\u0630\u0627 \u0627\u0644\u0645\u0648\u0636\u0648\u0639. (\u0647\u0630\u0627 \u0627\u0644\u062c\u0632\u0621 \u0645\u0627\u0632\u0627\u0644 \u0642\u064a\u062f \u0627\u0644\u062a\u0637\u0648\u064a\u0631)",
    }
    log_interaction(patient_id, raw_text, result)
    return result

if __name__ == "__main__":
    test_messages = [
        "\u0646\u062d\u0633 \u0628\u0648\u062c\u0639 \u0642\u0648\u064a \u0641\u064a \u0635\u062f\u0631\u064a \u0645\u0646 \u0634\u0648\u064a\u0629",
        "nsit nechreb ed-dwa lyoum",
        "\u0648\u0642\u062a\u0627\u0634 \u0644\u0627\u0632\u0645 \u0646\u0627\u062e\u0630 \u0627\u0644\u062f\u0648\u0627\u0621",
        "\u0633\u0644\u0627\u0645 \u0643\u064a\u0641\u0627\u0634 \u0627\u0644\u062d\u0627\u0644",
        "\u0646\u062d\u0633 \u0628\u0627\u0644\u0636\u064a\u0642 \u0645\u0646 \u0647\u0630\u0627 \u0627\u0644\u0645\u0631\u0636",
    ]
    for msg in test_messages:
        result = handle_message(msg)
        print(f"\nInput: {msg}")
        print(f"Route: {result['route']}")
        print(f"Response: {result['response']}")
